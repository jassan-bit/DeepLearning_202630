"""Admission-boundary and hash tests using artificial derivative files only."""
from contextlib import contextmanager
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_development_data as data
from e2_temporal_training import prepare_b


def synthetic_sources(admission="2023-09-23", *, corrupt_target=False):
    calendar=data.SOURCE_CALENDAR
    i=np.arange(len(calendar),dtype=float)
    close=100*np.exp(np.cumsum(.001+.005*np.sin(i/13)))
    x=np.column_stack((close*.999,close*1.002,close*.997,close,1000+i))
    x[calendar.get_loc(pd.Timestamp("2023-03-24",tz="UTC"))]=np.nan
    expected=pd.Series(np.log(x[:,3]),index=calendar).diff().rolling(7,min_periods=7).std(ddof=0).to_numpy()
    if corrupt_target:
        expected[calendar.get_loc(pd.Timestamp("2023-08-01",tz="UTC"))]+=.001
    streams={name:io.StringIO(newline="") for name in data.EXPECTED_SOURCES}
    writers={name:csv.writer(stream,lineterminator="\n") for name,stream in streams.items()}
    for name in writers:
        writers[name].writerow(data.HEADERS[name])
    cutoff=pd.Timestamp(admission,tz="UTC")
    for j,stamp in enumerate(calendar):
        excluded=stamp>cutoff
        market=["NOT_A_NUMBER"]*5 if excluded else x[j].tolist()
        target="NOT_A_NUMBER" if excluded else float(expected[j])
        writers[data.MARKET_PATH].writerow([str(stamp),*market,0])
        writers[data.TARGET_PATH].writerow([str(stamp),target,0,0,0])
    return {name:stream.getvalue().encode() for name,stream in streams.items()}


@contextmanager
def registered_cache(admission="2023-09-23", *, corrupt_target=False):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        raw=synthetic_sources(admission,corrupt_target=corrupt_target)
        hashes={name:hashlib.sha256(value).hexdigest() for name,value in raw.items()}
        protocol=json.dumps({"sources":hashes},sort_keys=True).encode()
        for name,value in {**raw,data.PROTOCOL_PATH:protocol}.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        with mock.patch.object(data,"EXPECTED_SOURCES",hashes), \
             mock.patch.object(data,"EXPECTED_PROTOCOL_SHA256",hashlib.sha256(protocol).hexdigest()):
            yield root,raw


class DevelopmentAdmissionTests(unittest.TestCase):
    def test_noise_admission_never_decodes_later_values_and_appends_unavailable_rows(self):
        with registered_cache() as (root,raw):
            panel,receipt=data.load_development(root)
            self.assertEqual(receipt["decoded_calendar_rows"],266)
            self.assertEqual(receipt["placeholder_rows"],8)
            self.assertEqual(str(panel.calendar[-1].date()),"2023-10-01")
            self.assertTrue(np.isnan(panel.targets[-8:]).all())
            self.assertTrue(np.isnan(panel.features[-8:]).all())
            self.assertFalse(panel.quality[-8:].any())
            self.assertFalse(receipt["parsing"][data.TARGET_PATH]["numerical_values_after_admission_decoded"])
            b=prepare_b(panel,"2023-10-01","2023-10-01")
            self.assertEqual((len(b.inner_train.dates),len(b.inner_validation.dates),len(b.refit_train.dates)),(161,83,251))
            self.assertEqual(sum(b.outer_plan.eligible),0)
            for path,expected in raw.items():
                self.assertEqual((root/path).read_bytes(),expected)

    def test_power_admission_stops_at_dec30_without_decoding_dec31_values(self):
        with registered_cache("2024-12-30") as (root,_):
            panel,receipt=data.load_development(root,"2024-12-30")
            self.assertEqual(receipt["decoded_calendar_rows"],730)
            self.assertEqual(receipt["placeholder_rows"],0)
            self.assertEqual(str(panel.calendar[-1].date()),"2024-12-30")
            b=prepare_b(panel,"2024-11-08","2024-12-30")
            self.assertEqual((len(b.inner_train.dates),len(b.inner_validation.dates),len(b.refit_train.dates)),(565,83,655))
            self.assertEqual(sum(b.outer_plan.eligible),53)

    def test_hash_changes_are_rejected_before_numeric_parsing(self):
        with registered_cache() as (root,_):
            path=root/data.TARGET_PATH
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                data.load_development(root)
        with registered_cache() as (root,_):
            path=root/data.PROTOCOL_PATH
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                data.load_development(root)

    def test_protected_or_unregistered_admission_fails_before_file_access(self):
        for end in ("2025-01-01","2024-12-31","2023-10-01","2026-08-31",None):
            with mock.patch.object(Path,"read_bytes",side_effect=AssertionError("unexpected source read")):
                with self.subTest(end=end),self.assertRaises(ValueError):
                    data.load_development("irrelevant",end)

    def test_label_reconstruction_rejects_a_hash_matched_wrong_target(self):
        with registered_cache(corrupt_target=True) as (root,_):
            with self.assertRaisesRegex(ValueError,"population SD"):
                data.load_development(root)

    def test_date_gaps_duplicates_and_naive_timestamps_are_rejected(self):
        raw=synthetic_sources()[data.TARGET_PATH]
        lines=raw.splitlines()
        bad_gap=b"\n".join(lines[:100]+lines[101:])+b"\n"
        bad_duplicate=b"\n".join(lines[:100]+[lines[99]]+lines[101:])+b"\n"
        for value in (bad_gap,bad_duplicate):
            with self.assertRaisesRegex(ValueError,"complete 2023--2024"):
                data._prefix_values(value,data.TARGET_PATH,pd.Timestamp("2023-09-23",tz="UTC"))
        naive=raw.replace(b"2023-01-01 00:00:00+00:00",b"2023-01-01 00:00:00",1)
        with self.assertRaisesRegex(ValueError,"UTC"):
            data._prefix_values(naive,data.TARGET_PATH,pd.Timestamp("2023-09-23",tz="UTC"))

    def test_registered_schema_and_protocol_source_mapping_cannot_be_swapped(self):
        with registered_cache() as (root,_):
            protocol={"sources":{**data.EXPECTED_SOURCES,data.TARGET_PATH:"0"*64}}
            encoded=json.dumps(protocol).encode()
            (root/data.PROTOCOL_PATH).write_bytes(encoded)
            with mock.patch.object(data,"EXPECTED_PROTOCOL_SHA256",hashlib.sha256(encoded).hexdigest()):
                with self.assertRaisesRegex(ValueError,"does not identify"):
                    data.load_development(root)
        raw=synthetic_sources()[data.TARGET_PATH].replace(b"sigma_7",b"RV_7",1)
        with self.assertRaisesRegex(ValueError,"columns changed"):
            data._prefix_values(raw,data.TARGET_PATH,pd.Timestamp("2023-09-23",tz="UTC"))


if __name__=="__main__":
    unittest.main()

