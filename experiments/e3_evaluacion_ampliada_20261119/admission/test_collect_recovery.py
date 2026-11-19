"""Offline synthetic fixtures for strict monthly-to-daily recovery.

The August fixture deliberately reproduces 27 valid observations on August
1..27, 2023, and four absent monthly days. ALL prices/volumes in these tests
are fabricated test data; they are not recovered market observations.
"""
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("recovery_under_test", HERE / "collect_recovery.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
base = recovery.base
SOURCE = "a" * 40
MONTH = "2023-08"
SYMBOL = "BTCUSD_PERP"
MONTH_NAME = SYMBOL + "-1d-" + MONTH + ".zip"
SPEC = {**base.DATASETS[1], "start": MONTH + "-01", "end": MONTH + "-31"}
HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time",
          "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def row(day, *, microseconds=False):
    unit = 1_000_000 if microseconds else 1000
    timestamp = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()) * unit
    return [str(timestamp), "10.0", "12.0", "9.0", "11.0", "100.0",
            str(timestamp + 86400 * unit - 1), "1", "5", "2", "3", "0"]


def archive(filename, rows, *, header=True):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    if header:
        writer.writerow(HEADER)
    writer.writerows(rows)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr(filename.removesuffix(".zip") + ".csv", text.getvalue())
    return data.getvalue()


class Response:
    status = 200
    headers = {"Content-Type": "application/zip", "ETag": "offline-test-fixture"}

    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, maximum):
        return self.raw[:maximum]


class Fixture:
    def __init__(self, month_rows=None, *, daily_rows=None):
        self.urls, self.calls = {}, []
        self.days = base.days_between(MONTH + "-01", MONTH + "-31")
        self.month_rows = [row(day) for day in self.days[:27]] if month_rows is None else month_rows
        self.month_bytes = archive(MONTH_NAME, self.month_rows)
        self.add(SPEC["prefix"], MONTH_NAME, self.month_bytes)
        self.daily_prefix = SPEC["prefix"].replace("/monthly/", "/daily/")
        for day in self.days:
            filename = SYMBOL + "-1d-" + day + ".zip"
            rows = daily_rows.get(day, [row(day)]) if daily_rows is not None else [row(day)]
            self.add(self.daily_prefix, filename, archive(filename, rows))

    def add(self, prefix, name, data):
        self.urls[prefix + name] = data
        self.urls[prefix + name + ".CHECKSUM"] = (base.sha(data) + "  " + name + "\n").encode()

    def open(self, request, timeout):
        self.calls.append(request.full_url)
        if request.full_url not in self.urls:
            raise urllib.error.URLError("offline synthetic absent file")
        return Response(self.urls[request.full_url])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "data"

    def tearDown(self):
        self.temp.cleanup()

    def admit(self, fixture):
        return recovery.admit_vision(SPEC, self.output, fixture.open, 0)

    def no_admission(self):
        self.assertFalse((self.output / "datasets/D03.csv").exists())
        self.assertFalse((self.output / "responses/D03/download_receipts.json").exists())

    def test_august_27_day_fixture_reproduces_original_exact_error(self):
        fixture = Fixture()
        with self.assertRaisesRegex(base.AdmissionError, "monthly_calendar_order_duplicates_or_missing_days"):
            base.parse_month(fixture.month_bytes, MONTH_NAME, MONTH)
        records, parsed = recovery._parse_rows(fixture.month_bytes, MONTH_NAME, MONTH)
        self.assertEqual(parsed["rows"], 27)
        self.assertEqual([r["date"] for r in records], fixture.days[:27])
        self.assertEqual(fixture.days[27:], ["2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31"])

    def test_entire_month_daily_acquisition_preserves_monthly_bytes_and_all_rows(self):
        fixture = Fixture()
        report = self.admit(fixture)
        self.assertEqual(report["rows"], 31)
        self.assertEqual(report["rows_filled"], 0)
        self.assertEqual(report["rows_removed"], 0)
        recovery_receipt = report["monthly_sources"][0]["recovery"]
        self.assertEqual(recovery_receipt["daily_rows"], 31)
        self.assertEqual(len(recovery_receipt["daily_sources"]), 31)
        self.assertEqual(len(recovery_receipt["overlap_dates"]), 27)
        self.assertIs(recovery_receipt["overlap_OHLCV_exact_decimal_equality"], True)
        self.assertEqual(len(fixture.calls), 64)  # 2 monthly + 31 daily pairs
        self.assertEqual((self.output / "responses/D03" / MONTH_NAME).read_bytes(), fixture.month_bytes)
        saved = list(csv.DictReader(io.StringIO((self.output / "datasets/D03.csv").read_text())))
        self.assertEqual([r["date"] for r in saved], fixture.days)

    def test_daily_numerically_equal_decimal_formats_are_accepted_without_tolerance(self):
        different_format = row("2023-08-01")
        different_format[1:6] = ["10", "12", "9", "11", "100"]
        report = self.admit(Fixture(daily_rows={"2023-08-01": [different_format]}))
        self.assertEqual(report["rows"], 31)

    def test_generic_internal_missing_dates_and_microsecond_daily_clocks(self):
        days = base.days_between(MONTH + "-01", MONTH + "-31")
        monthly = [row(day) for index, day in enumerate(days) if index not in {1, 8, 15, 22}]
        fixture = Fixture(month_rows=monthly, daily_rows={day:[row(day, microseconds=True)] for day in days})
        report = self.admit(fixture)
        recovered = report["monthly_sources"][0]["recovery"]
        self.assertEqual(recovered["missing_monthly_dates"], [days[i] for i in (1, 8, 15, 22)])
        self.assertEqual(recovered["daily_rows"], 31)
        self.assertEqual(recovered["daily_sources"][0]["parsed"]["timestamp_units"], ["microseconds"])

    def test_daily_overlap_price_or_volume_change_fails_and_rolls_back_admission(self):
        for column, value in ((1, "10.0000000000000000000001"), (5, "101")):
            with self.subTest(column=column):
                self.output = Path(self.temp.name) / str(column)
                changed = row("2023-08-01"); changed[column] = value
                fixture = Fixture(daily_rows={"2023-08-01": [changed]})
                with self.assertRaisesRegex(base.AdmissionError, "monthly_daily_overlap_OHLCV_mismatch:2023-08-01"):
                    self.admit(fixture)
                self.no_admission()
                attempt = json.loads((self.output / "responses/D03/recovery_attempts.json").read_bytes())[MONTH]
                self.assertEqual(attempt["status"], "blocked")
                self.assertTrue((self.output / "responses/D03" / MONTH_NAME).exists())

    def test_absent_daily_archive_fails_closed_but_retains_raw_progress(self):
        fixture = Fixture()
        del fixture.urls[fixture.daily_prefix + SYMBOL + "-1d-2023-08-28.zip"]
        with self.assertRaisesRegex(base.AdmissionError, "source_unavailable"):
            self.admit(fixture)
        self.no_admission()
        receipts = json.loads((self.output / "responses/D03/daily_recovery/2023-08/download_receipts.json").read_bytes())
        self.assertEqual(len(receipts), 27)
        self.assertTrue((self.output / "responses/D03" / MONTH_NAME).exists())

    def test_missing_day_retry_reuses_all_verified_bytes_and_first_http_receipts(self):
        fixture = Fixture()
        missing = fixture.daily_prefix + SYMBOL + "-1d-2023-08-28.zip"
        original = fixture.urls.pop(missing)
        with self.assertRaises(base.AdmissionError): self.admit(fixture)
        prior = (self.output / "responses/D03/daily_recovery/2023-08/download_receipts.json").read_bytes()
        fixture.calls.clear(); fixture.urls[missing] = original
        report = self.admit(fixture)
        self.assertEqual(report["rows"], 31)
        self.assertEqual(len(fixture.calls), 7)  # day28 ZIP, then3 complete pairs
        all_receipts = json.loads((self.output / "responses/D03/daily_recovery/2023-08/download_receipts.json").read_bytes())
        for day, receipt in json.loads(prior).items(): self.assertEqual(all_receipts[day], receipt)

    def test_checksum_mismatch_never_recovers_month_and_daily_mismatch_never_admits(self):
        for kind in ("monthly", "daily"):
            with self.subTest(kind=kind):
                self.output = Path(self.temp.name) / kind
                fixture = Fixture()
                prefix = SPEC["prefix"] if kind == "monthly" else fixture.daily_prefix
                filename = MONTH_NAME if kind == "monthly" else SYMBOL + "-1d-2023-08-01.zip"
                fixture.urls[prefix + filename + ".CHECKSUM"] = ("0"*64 + "  " + filename + "\n").encode()
                with self.assertRaisesRegex(base.AdmissionError, "official_archive_checksum_mismatch"):
                    self.admit(fixture)
                self.no_admission()
                if kind == "monthly": self.assertEqual(len(fixture.calls), 2)

    def test_invalid_duplicate_unordered_empty_months_have_no_daily_fallback(self):
        days = base.days_between(MONTH + "-01", MONTH + "-31")
        invalid = [[], [row(days[0]), row(days[0])], [row(days[1]), row(days[0])]]
        bad_price = row(days[0]); bad_price[2] = "1"
        invalid.append([bad_price])
        bad_time = row(days[0]); bad_time[6] = str(int(bad_time[6]) - 1)
        invalid.append([bad_time])
        outside = row("2023-09-01"); invalid.append([outside])
        fractional = row(days[0]); fractional[5] = "10.5"; invalid.append([fractional])
        for index, rows in enumerate(invalid):
            with self.subTest(index=index):
                self.output = Path(self.temp.name) / str(index)
                fixture = Fixture(month_rows=rows)
                with self.assertRaises(base.AdmissionError): self.admit(fixture)
                self.assertEqual(len(fixture.calls), 2)
                self.no_admission()

    def test_daily_wrong_date_duplicate_invalid_timestamp_and_contracts_rejected(self):
        malformed = [[row("2023-08-02")], [row("2023-08-01"), row("2023-08-01")]]
        timestamp = row("2023-08-01"); timestamp[0] = str(int(timestamp[0]) + 1); malformed.append([timestamp])
        fractional = row("2023-08-01"); fractional[5] = "0.5"; malformed.append([fractional])
        for index, rows in enumerate(malformed):
            with self.subTest(index=index):
                self.output = Path(self.temp.name) / str(index)
                with self.assertRaises(base.AdmissionError):
                    self.admit(Fixture(daily_rows={"2023-08-01": rows}))
                self.no_admission()

    def test_complete_valid_month_matches_historical_parsed_and_entire_receipt(self):
        fixture = Fixture(month_rows=[row(d) for d in base.days_between(MONTH+"-01", MONTH+"-31")])
        historical = base.admit_vision(SPEC, self.output, fixture.open, 0)
        receipt_path = self.output / "responses/D03/download_receipts.json"
        prior = receipt_path.read_bytes()
        expected_csv = (self.output / "datasets/D03.csv").read_bytes()
        fixture.calls.clear()
        result = self.admit(fixture)
        self.assertEqual(result, historical)
        self.assertEqual(receipt_path.read_bytes(), prior)
        self.assertEqual((self.output / "datasets/D03.csv").read_bytes(), expected_csv)
        self.assertEqual(fixture.calls, [])
        self.assertNotIn("recovery", result["monthly_sources"][0])

    def test_successful_recovery_is_idempotent_and_does_no_network_on_retry(self):
        fixture = Fixture(); first = self.admit(fixture)
        before = {str(p):p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        fixture.calls.clear()
        self.assertEqual(self.admit(fixture), first)
        self.assertEqual(fixture.calls, [])
        self.assertEqual(before, {str(p):p.read_bytes() for p in self.output.rglob("*") if p.is_file()})

    def test_changed_cached_archive_or_receipt_is_rejected(self):
        for kind in ("archive", "receipt"):
            with self.subTest(kind=kind):
                self.output = Path(self.temp.name) / kind
                fixture=Fixture(); self.admit(fixture)
                if kind == "archive":
                    path = self.output / "responses/D03/daily_recovery/2023-08" / (SYMBOL+"-1d-2023-08-01.zip")
                    path.write_bytes(path.read_bytes()+b"changed")
                else:
                    path=self.output / "responses/D03/download_receipts.json"
                    receipts=json.loads(path.read_bytes());receipts[MONTH]["recovery"]["overlap_dates"]=[]
                    path.write_bytes(base.canonical(receipts))
                with self.assertRaises(base.AdmissionError): self.admit(fixture)

    def test_run_binds_both_collectors_source_documents_and_preserves_original_config(self):
        fixture = Fixture(); self.output.mkdir()
        original = base.canonical({"historical_config": True, "source_commit": "b"*40})
        (self.output / "run_config.json").write_bytes(original)
        d01 = {"id":"D01", "admitted":False, "target_values_computed":False}
        with patch.object(base, "DATASETS", (SPEC,)), patch.object(base, "admit_d01", return_value=d01):
            report = recovery.run(self.output, SOURCE, opener=fixture.open, pause_seconds=0)
            before = {str(p):p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
            fixture.calls.clear()
            self.assertEqual(recovery.run(self.output,SOURCE,opener=fixture.open,pause_seconds=0), report)
        self.assertEqual((self.output / "run_config.json").read_bytes(), original)
        self.assertEqual(report["collector_components"], recovery.collector_components())
        self.assertEqual(report["collector_sha256"], base.sha(base.canonical(recovery.collector_components())))
        self.assertEqual(fixture.calls, [])
        self.assertEqual(before, {str(p):p.read_bytes() for p in self.output.rglob("*") if p.is_file()})
        for document in report["source_documents"]:
            self.assertEqual(base.sha((self.output / document["path"]).read_bytes()), document["sha256"])
        for key in ("real_campaign_allowed", "target_values_computed", "protocol_frozen"):
            self.assertIs(report[key], False)
        self.assertEqual(report["new_real_data_fits"], 0)

    def test_run_persists_blocked_report_and_raw_manifest_without_admitting_missing_day(self):
        fixture=Fixture()
        del fixture.urls[fixture.daily_prefix + SYMBOL + "-1d-2023-08-28.zip.CHECKSUM"]
        with patch.object(base,"DATASETS",(SPEC,)), patch.object(base,"admit_d01",return_value={"id":"D01","admitted":False}):
            report=recovery.run(self.output,SOURCE,opener=fixture.open,pause_seconds=0)
        self.assertEqual(report["datasets_admitted"],0)
        d03=report["datasets"][-1]
        self.assertIs(d03["admitted"],False)
        self.assertIn("source_unavailable",d03["error"])
        self.assertIs(d03["missing_values_filled"],False)
        self.no_admission()
        manifest=json.loads((self.output/"raw_manifest.json").read_bytes())
        self.assertIn("responses/D03/"+MONTH_NAME,manifest["files"])
        self.assertNotIn("datasets/D03.csv",manifest["files"])


if __name__ == "__main__":
    unittest.main()
