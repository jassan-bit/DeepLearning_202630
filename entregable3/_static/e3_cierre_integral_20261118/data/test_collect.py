"""Meaningful offline admission contracts; fixtures are artificial, not market evidence."""
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


spec = importlib.util.spec_from_file_location("e3_data_collect", Path(__file__).with_name("collect.py"))
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def artificial_month(month="2020-02", *, header=False, microseconds=False, mutate=None):
    labels = collect.days_between(month + "-01", month + "-29")
    unit = 1_000_000 if microseconds else 1_000
    records = []
    for label in labels:
        opened = int(datetime.combine(date.fromisoformat(label), datetime.min.time(), tzinfo=timezone.utc).timestamp()) * unit
        records.append([str(opened), "100", "105", "97", "102", "4", str(opened + 86400*unit - 1), "2", "3", "2", "1", "0"])
    if mutate:
        mutate(records)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    if header:
        writer.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time", "base_asset_volume", "count", "taker_buy_volume", "taker_buy_base_volume", "ignore"])
    writer.writerows(records)
    raw = io.BytesIO()
    filename = "FIXTURE-1d-" + month + ".zip"
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename.removesuffix(".zip") + ".csv", buffer.getvalue())
    return filename, raw.getvalue()


class Response:
    def __init__(self, body):
        self.body, self.status, self.headers = body, 200, {"ETag": "fixture-etag"}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, limit):
        return self.body[:limit]


class AdmissionContracts(unittest.TestCase):
    def test_monthly_sources_and_exact_expected_sizes(self):
        a, b = collect.DATASETS
        self.assertEqual(len(collect.months_between(a["start"], a["end"])), 48)
        self.assertEqual(len(collect.months_between(b["start"], b["end"])), 52)
        self.assertEqual(len(collect.days_between(a["start"], a["end"])), 1461)
        self.assertEqual(len(collect.days_between(b["start"], b["end"])), 1583)
        self.assertEqual(b["volume_units"], "contracts")

    def test_leap_calendar_header_and_timestamp_units(self):
        for header, microseconds in [(False, False), (True, False), (True, True)]:
            name, raw = artificial_month(header=header, microseconds=microseconds)
            rows, evidence = collect.parse_month(raw, name, "2020-02")
            self.assertEqual(len(rows), 29)
            self.assertEqual(rows[-1]["date"], "2020-02-29")
            self.assertEqual(evidence["timestamp_units"], ["microseconds" if microseconds else "milliseconds"])

    def test_missing_duplicate_reordered_or_shifted_clock_rejected(self):
        mutations = [lambda r:r.pop(4), lambda r:r.insert(4, r[4]),
                     lambda r:r.reverse(), lambda r:r[0].__setitem__(0,str(int(r[0][0])+1000)),
                     lambda r:r[0].__setitem__(6,str(int(r[0][6])-1))]
        for mutation in mutations:
            name, raw = artificial_month(mutate=mutation)
            with self.assertRaises(collect.AdmissionError):
                collect.parse_month(raw, name, "2020-02")

    def test_nonfinite_incoherent_and_negative_ohlcv_rejected(self):
        for index, value in [(1, "NaN"), (2, "101"), (3, "103"), (4, "0"), (5, "-1"), (8, "1.2")]:
            name, raw = artificial_month(mutate=lambda rows,i=index,v=value:rows[0].__setitem__(i,v))
            with self.assertRaises(collect.AdmissionError):
                collect.parse_month(raw, name, "2020-02")

    def test_checksum_file_identity_and_body_are_both_bound(self):
        name, raw = artificial_month()
        body = (hashlib.sha256(raw).hexdigest() + "  " + name + "\n").encode()
        self.assertEqual(collect.checksum_digest(body, name), hashlib.sha256(raw).hexdigest())
        with self.assertRaises(collect.AdmissionError):
            collect.checksum_digest(body, "another.zip")
        with self.assertRaises(collect.AdmissionError):
            collect.checksum_digest(b"<html>challenge</html>", name)
        fixture = {"id":"D02", "symbol":"FIXTURE", "market":"spot", "start":"2020-02-01", "end":"2020-02-29", "volume_units":"fixture", "product":"artificial", "prefix":"https://example.invalid/"}
        def opener(req, timeout):
            return Response(body if req.full_url.endswith(".CHECKSUM") else raw + b"corruption")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(collect.AdmissionError, "checksum_mismatch"):
                collect.admit_vision(fixture, Path(tmp), opener, 0)
            self.assertFalse((Path(tmp)/"datasets/D02.csv").exists())

    def test_cached_retry_never_refetches_or_rewrites_first_receipt(self):
        name, raw = artificial_month()
        checksum = (collect.sha(raw) + "  " + name + "\n").encode()
        fixture = {"id":"D02", "symbol":"FIXTURE", "market":"spot", "start":"2020-02-01", "end":"2020-02-29", "volume_units":"fixture", "product":"artificial", "prefix":"https://example.invalid/"}
        calls = []
        def opener(req, timeout):
            calls.append(req.full_url)
            return Response(checksum if req.full_url.endswith(".CHECKSUM") else raw)
        def unavailable(req, timeout):
            raise AssertionError("A verified retry must not use the network")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = collect.admit_vision(fixture, out, opener, 0)
            receipt_path = out/"responses/D02/download_receipts.json"
            saved = receipt_path.read_bytes()
            second = collect.admit_vision(fixture, out, unavailable, 0)
            self.assertEqual(first, second)
            self.assertEqual(receipt_path.read_bytes(), saved)
            self.assertEqual(len(calls), 2)

    def test_coin_m_contracts_are_not_silently_replaced_by_fractional_base_volume(self):
        name, raw = artificial_month(mutate=lambda rows:rows[0].__setitem__(5,"0.125"))
        checksum = (collect.sha(raw) + "  " + name + "\n").encode()
        fixture = {"id":"D03", "symbol":"FIXTURE", "market":"coin_margined_perpetual", "start":"2020-02-01", "end":"2020-02-29", "volume_units":"contracts", "product":"artificial", "prefix":"https://example.invalid/"}
        def opener(req, timeout):
            return Response(checksum if req.full_url.endswith(".CHECKSUM") else raw)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(collect.AdmissionError, "contract_volume_not_integral"):
                collect.admit_vision(fixture, Path(tmp), opener, 0)
            self.assertFalse((Path(tmp)/"datasets/D03.csv").exists())

    def test_existing_D01_is_normalized_without_adjustment_or_target_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = collect.admit_d01(Path(tmp))
            self.assertEqual(result["rows"], 3270)
            self.assertEqual(result["possible_window_target_pairs"], 3263)
            self.assertFalse(result["target_values_computed"])
            with (Path(tmp)/result["path"]).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), list(collect.FIELDS))
            self.assertNotIn("Adj Close", rows[0])
            self.assertEqual(rows[0]["date"], "2000-03-01")

    def test_unavailable_sources_never_become_admissions_or_predictions(self):
        def unavailable(req, timeout):
            raise collect.urllib.error.URLError("artificial_transport_failure")
        with tempfile.TemporaryDirectory() as tmp:
            report = collect.run(Path(tmp)/"run", "a"*40, opener=unavailable, pause_seconds=0)
            self.assertEqual(report["datasets_admitted"], 1)
            self.assertEqual([r["admitted"] for r in report["datasets"]], [True,False,False])
            self.assertFalse(report["real_campaign_allowed"])
            self.assertEqual(report["new_real_data_fits"], 0)
            self.assertEqual(report["new_market_predictions"], 0)
            self.assertFalse(report["target_values_computed"])
            self.assertFalse((Path(tmp)/"run/datasets/D02.csv").exists())
            self.assertFalse((Path(tmp)/"run/datasets/D03.csv").exists())
            with self.assertRaisesRegex(collect.AdmissionError, "cached_source_changed"):
                collect.run(Path(tmp)/"run", "b"*40, opener=unavailable, pause_seconds=0)


if __name__ == "__main__":
    unittest.main()
