"""Software-only regression tests, using the real header and generated rows.

The diagnostic receipt binds the header to a provider-checksummed real archive;
none of its OHLCV values are copied into these generated fixtures or published.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cm_header_addon',HERE/'cm_header_parser.py')
addon=importlib.util.module_from_spec(spec);spec.loader.exec_module(addon)


def source_path():
    paths=[Path(os.environ['CM_REPAIR_SOURCE_DATA'])] if os.environ.get('CM_REPAIR_SOURCE_DATA') else []
    if len(HERE.parents)>2:
        paths.append(HERE.parents[2]/'experiments/e3_hypothesis_adequacy_20261122/runtime/engine/data.py')
    paths.append(Path('/workspace/e3_hypothesis_adequacy_20261122/runtime/engine/data.py'))
    return next(path for path in paths if path.is_file())


def load_source():
    spec=importlib.util.spec_from_file_location('s2_parser_for_software_fixture',source_path())
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    return source


def generated_archive(header=addon.LEGACY_CM_HEADER, *, rows=31, newline='\n', bom=False, mutate=None, repeated=False):
    buffer=io.StringIO(newline='')
    writer=csv.writer(buffer,lineterminator=newline)
    if header is not None:writer.writerow(header)
    for i in range(rows):
        date=dt.datetime(2022,7,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=i)
        stamp=int(date.timestamp())*1000
        row=[stamp,100+i,110+i,90+i,101+i,12345+i,stamp+86400000-1,2.5+i/100,10,2,1,0]
        if mutate:mutate(row,i)
        writer.writerow(row)
        if repeated and i==3:writer.writerow(addon.LEGACY_CM_HEADER)
    text=buffer.getvalue()
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as archive:
        entry=zipfile.ZipInfo('GENERATED_SOFTWARE_FIXTURE.csv',date_time=(1980,1,1,0,0,0))
        archive.writestr(entry,(b'\xef\xbb\xbf' if bom else b'')+text.encode())
    return out.getvalue()


def arrays_equal(test,a,b):
    test.assertEqual(a['provider'],b['provider'])
    test.assertEqual(a['metadata'],b['metadata'])
    for key in ('dates','ohlcv','split'):np.testing.assert_array_equal(a[key],b[key])


class CMHeaderTests(unittest.TestCase):
    def setUp(self):
        self.source=load_source()
        self.original=self.source.parse_binance_zip
        self.parser=addon.bind_parser(self.source)
        self.acquired='2022-08-03T00:00:00+00:00'

    def test_exact_real_july_header_is_bound_by_primary_diagnostic(self):
        receipt=json.loads((HERE.parent/'REAL_CM_HEADER_DIAGNOSTIC.json').read_text())
        row=next(r for r in receipt['archives'] if r['archive_name']=='BTCUSD_PERP-1d-2022-07.zip')
        self.assertEqual(tuple(row['header']),addon.LEGACY_CM_HEADER)
        self.assertEqual(row['archive_sha256'],'2b2c4aa1bb3a312c17c61fb4b7748f137e19eeed9a3c8834a2a6556d1bbaa017')
        self.assertEqual(row['bar_count'],31)
        self.assertTrue(row['S2_rejects_column7_name'])
        self.assertFalse(receipt['raw_prices_or_volumes_published'])

    def test_s2_rejects_real_header_but_repair_preserves_physical_base_column7(self):
        raw=generated_archive()
        with self.assertRaisesRegex(self.source.AdmissionError,'Wrong market-volume header'):
            self.original(raw,acquired_utc=self.acquired,market_type='cm')
        got=self.parser(raw,acquired_utc=self.acquired,market_type='cm')
        canonical=list(addon.LEGACY_CM_HEADER);canonical[7]='base_volume'
        expected=self.original(generated_archive(canonical),acquired_utc=self.acquired,market_type='cm')
        arrays_equal(self,got,expected)
        self.assertEqual(len(got['dates']),31)
        self.assertEqual(got['ohlcv'][0,4],2.5)
        self.assertNotEqual(got['ohlcv'][0,4],12345)
        self.assertEqual(got['metadata']['volume_column_zero_based'],7)
        self.assertEqual(got['metadata']['parser_version'],'crypto-markets-bounded-archives-base-volume-v2.1')

    def test_canonical_header_and_no_header_remain_identical(self):
        canonical=list(addon.LEGACY_CM_HEADER);canonical[7]='base_asset_volume'
        for header in (None,canonical):
            with self.subTest(header=header):
                raw=generated_archive(header)
                arrays_equal(self,self.parser(raw,acquired_utc=self.acquired,market_type='cm'),
                    self.original(raw,acquired_utc=self.acquired,market_type='cm'))
        self.assertEqual(self.parser.__cm_header_repair_events__,[])

    def test_spot_and_usdm_delegation_never_treats_column7_as_selected_volume(self):
        raw=generated_archive()
        for market in ('spot','um'):
            with self.subTest(market=market):
                actual=self.parser(raw,acquired_utc=self.acquired,market_type=market)
                arrays_equal(self,actual,self.original(raw,acquired_utc=self.acquired,market_type=market))
                self.assertEqual(actual['ohlcv'][0,4],12345)
        self.assertEqual(self.parser.__cm_header_repair_events__,[])

    def test_unknown_generic_header_variants_are_not_accepted(self):
        variants=[]
        for column,text in ((7,'quote_asset_volume'),(7,'some_volume'),(8,'trade_count'),(10,'taker_buy_base_volume')):
            header=list(addon.LEGACY_CM_HEADER);header[column]=text;variants.append(header)
        swapped=list(addon.LEGACY_CM_HEADER);swapped[1],swapped[2]=swapped[2],swapped[1];variants.append(swapped)
        for header in variants:
            with self.subTest(header=header):
                with self.assertRaises(self.source.AdmissionError):
                    self.parser(generated_archive(header),acquired_utc=self.acquired,market_type='cm')
        self.assertEqual(self.parser.__cm_header_repair_events__,[])

    def test_invalid_values_and_temporal_checks_are_not_relaxed(self):
        cases={
            'negative_base':lambda row,i:row.__setitem__(7,-1),
            'nan_base':lambda row,i:row.__setitem__(7,'nan'),
            'negative_contracts':lambda row,i:row.__setitem__(5,-1),
            'nan_contracts':lambda row,i:row.__setitem__(5,'nan'),
            'wrong_close':lambda row,i:row.__setitem__(6,int(row[6])+1),
            'non_midnight':lambda row,i:row.__setitem__(0,int(row[0])+1000),
        }
        for name,mutate in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(self.source.AdmissionError):
                    self.parser(generated_archive(mutate=mutate),acquired_utc=self.acquired,market_type='cm')
        with self.assertRaisesRegex(self.source.AdmissionError,'Incomplete'):
            self.parser(generated_archive(rows=1),acquired_utc='2022-07-01T12:00:00+00:00',market_type='cm')
        self.assertEqual(self.parser.__cm_header_repair_events__,[])

    def test_repeated_header_and_wrong_column_count_still_fail(self):
        with self.assertRaises((ValueError,self.source.AdmissionError)):
            self.parser(generated_archive(repeated=True),acquired_utc=self.acquired,market_type='cm')
        with self.assertRaisesRegex(self.source.AdmissionError,'schema'):
            self.parser(generated_archive(addon.LEGACY_CM_HEADER[:-1]),acquired_utc=self.acquired,market_type='cm')

    def test_in_memory_adaptation_preserves_all_data_lines_and_input_archive_bytes(self):
        for newline,bom in (('\n',False),('\r\n',False),('\r\n',True)):
            with self.subTest(newline=newline,bom=bom):
                raw=generated_archive(newline=newline,bom=bom)
                digest=hashlib.sha256(raw).hexdigest()
                adapted=addon._adapt_exact_header(raw)
                with zipfile.ZipFile(io.BytesIO(raw)) as a,zipfile.ZipFile(io.BytesIO(adapted)) as b:
                    original_csv=a.read(a.namelist()[0]).decode('utf-8-sig')
                    changed_csv=b.read(b.namelist()[0]).decode()
                self.assertEqual(original_csv.split('\n',1)[1],changed_csv.split('\n',1)[1])
                self.parser(raw,acquired_utc=self.acquired,market_type='cm')
                self.assertEqual(hashlib.sha256(raw).hexdigest(),digest)

    def test_monthly_calendar_is_still_checked_by_frozen_collector(self):
        self.source.parse_binance_zip=self.parser
        item={'first_session':'2022-07-01','last_session':'2022-07-31'}
        with self.assertRaisesRegex(self.source.AdmissionError,'unexpected/missing/out-of-range'):
            self.source._validate_binance_archive(generated_archive(rows=30),item,'binance_vision_cm',self.acquired)

    def test_binding_is_explicit_idempotent_and_source_hash_checked(self):
        self.assertIs(self.source.parse_binance_zip,self.original)
        self.source.parse_binance_zip=self.parser
        self.assertIs(addon.bind_parser(self.source),self.parser)
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'data.py';bad.write_text('# not S2')
            with self.assertRaisesRegex(RuntimeError,'exact immutable S2'):
                addon.bind_parser(SimpleNamespace(__file__=str(bad)))

    def test_receipt_metadata_is_deduplicated_without_prices(self):
        raw=generated_archive()
        for _ in range(2):self.parser(raw,acquired_utc=self.acquired,market_type='cm')
        events=self.parser.__cm_header_repair_events__
        self.assertEqual(len(events),1)
        event=events[0]
        self.assertEqual(event['source_archive_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(event['observed_header'],list(addon.LEGACY_CM_HEADER))
        self.assertEqual(event['base_volume_column_zero_based'],7)
        self.assertFalse(event['raw_source_bytes_rewritten'])
        self.assertNotIn('ohlcv',event)


if __name__=='__main__':unittest.main(verbosity=2)
