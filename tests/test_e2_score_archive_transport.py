"""Synthetic transport and score-admission checks; no market payload downloads."""
from __future__ import annotations

import hashlib
from http.client import IncompleteRead
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

import e2_score_archive_transport as transport
import e2_score_source_worker as worker

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('e2_original_final_data_transport_test', ROOT/'scripts/e2_reduced_final_data.py')
frozen_data = importlib.util.module_from_spec(loader)
loader.loader.exec_module(frozen_data)


class Response:
    def __init__(self, raw, *, error=None, url='https://data.binance.vision/fixture.zip'):
        self.raw, self.error, self.url = raw, error, url
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def geturl(self): return self.url
    def read(self, size):
        if self.error is not None: raise self.error
        return self.raw[:size]


class ArchiveTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw_root = Path(self.temp.name)/'raw'
        self.payload = b'synthetic archive byte fixture; no market rows\n'
        self.record = {'archive':'BTCUSDT-1m-2025-01.zip',
            'relative_path':'BTCUSDT/1m/monthly/BTCUSDT-1m-2025-01.zip',
            'url':'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01.zip',
            'bytes':len(self.payload), 'sha256':hashlib.sha256(self.payload).hexdigest()}
        self.target = self.raw_root/self.record['relative_path']
        self.events, self.delays = [], []

    def wrap(self, download=None):
        with mock.patch.object(transport,'load_registered_archives',return_value=(dict(self.record),)):
            return transport.bounded_archive_transport(
                frozen_data._download_archive if download is None else download,
                ROOT,self.raw_root,sleep=self.delays.append,
                emit=lambda text,**kw:self.events.append(json.loads(text)))

    def test_actual_twenty_archive_allowlist_metadata_and_frozen_bytes(self):
        records = transport.load_registered_archives(ROOT)
        self.assertEqual(len(records),20)
        self.assertEqual(records[0]['archive'],'BTCUSDT-1m-2025-01.zip')
        self.assertEqual(records[-1]['archive'],'BTCUSDT-1m-2026-08.zip')
        self.assertEqual(hashlib.sha256((ROOT/worker.FINAL_DATA_PATH).read_bytes()).hexdigest(),worker.FINAL_DATA_SHA256)
        self.assertEqual(hashlib.sha256((ROOT/worker.AGGREGATOR_PATH).read_bytes()).hexdigest(),worker.AGGREGATOR_SHA256)
        self.assertEqual(hashlib.sha256((ROOT/worker.PRIOR_WORKER_PATH).read_bytes()).hexdigest(),worker.PRIOR_WORKER_SHA256)

    def test_body_read_and_open_transients_retry_only_transport_then_write_exact_bytes(self):
        opener=mock.Mock(side_effect=[
            Response(b'',error=URLError('temporary body interruption')),
            HTTPError(self.record['url'],503,'temporary',{},None),
            Response(self.payload)])
        wrapped=self.wrap()
        self.assertEqual(wrapped(self.record,self.target,opener),self.payload)
        self.assertEqual(opener.call_count,3)
        self.assertEqual(self.delays,[1.,2.])
        self.assertEqual(self.target.read_bytes(),self.payload)
        self.assertEqual([x['event'] for x in self.events],[
            'final_archive_transport_started','final_archive_transport_retry',
            'final_archive_transport_retry','final_archive_transport_completed'])
        self.assertTrue(all(x.get('scientific_retry') is False for x in self.events[1:]))

    def test_404_is_terminal_without_file_or_retry(self):
        opener=mock.Mock(side_effect=HTTPError(self.record['url'],404,'missing',{},None))
        with self.assertRaises(HTTPError):self.wrap()(self.record,self.target,opener)
        self.assertEqual(opener.call_count,1)
        self.assertEqual(self.delays,[])
        self.assertFalse(self.target.exists())

    def test_three_attempts_is_an_absolute_transport_bound(self):
        opener=mock.Mock(side_effect=URLError('temporary'))
        with self.assertRaises(URLError):self.wrap()(self.record,self.target,opener)
        self.assertEqual(opener.call_count,3)
        self.assertEqual(self.delays,[1.,2.])
        self.assertFalse(self.target.exists())

    def test_bad_payload_checksum_fails_closed_before_cache_creation(self):
        opener=mock.Mock(return_value=Response(b'x'*len(self.payload)))
        with self.assertRaisesRegex(ValueError,'pinned'):self.wrap()(self.record,self.target,opener)
        self.assertEqual(opener.call_count,1)
        self.assertEqual(self.delays,[])
        self.assertFalse(self.target.exists())

    def test_verified_cached_archive_is_reused_without_a_network_call(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.payload)
        opener=mock.Mock(side_effect=AssertionError('cached data must not redownload'))
        self.assertEqual(self.wrap()(self.record,self.target,opener),self.payload)
        opener.assert_not_called()
        self.assertTrue(self.events[-1]['cached'])

    def test_corrupted_cache_is_retained_and_never_retried(self):
        self.target.parent.mkdir(parents=True)
        corrupted=b'X'*len(self.payload)
        self.target.write_bytes(corrupted)
        opener=mock.Mock()
        with self.assertRaisesRegex(ValueError,'pinned'):self.wrap()(self.record,self.target,opener)
        opener.assert_not_called()
        self.assertEqual(self.target.read_bytes(),corrupted)
        self.assertEqual(self.delays,[])

    def test_unregistered_record_redirect_and_target_are_rejected(self):
        opener=mock.Mock(return_value=Response(self.payload,url='https://example.org/archive.zip'))
        with self.assertRaisesRegex(ValueError,'redirected'):self.wrap()(self.record,self.target,opener)
        self.assertFalse(self.target.exists())
        opener.reset_mock()
        changed={**self.record,'url':'https://example.org/archive.zip'}
        with self.assertRaisesRegex(ValueError,'unregistered'):self.wrap()(changed,self.target,opener)
        opener.assert_not_called()
        with self.assertRaisesRegex(ValueError,'path differs'):
            self.wrap()(self.record,self.raw_root/'elsewhere.zip',opener)
        opener.assert_not_called()

    def test_timeout_and_reset_retry_but_permission_errors_do_not(self):
        for error in (TimeoutError('timeout'),ConnectionResetError('reset'),IncompleteRead(b'partial',99)):
            with self.subTest(error=type(error).__name__):
                opener=mock.Mock(side_effect=[error,Response(self.payload)])
                self.assertEqual(self.wrap()(self.record,self.target,opener),self.payload)
                self.assertEqual(opener.call_count,2)
                self.target.unlink()
        opener=mock.Mock(side_effect=PermissionError('permission denied'))
        with self.assertRaises(PermissionError):self.wrap()(self.record,self.target,opener)
        self.assertEqual(opener.call_count,1)


class ScoreWorkerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.final=Path(self.temp.name)/'final'
        self.output=self.final/worker.OUTPUT_NAME
        self.output.mkdir(parents=True)
        self.path=self.output/'worker_spec.json'
        self.spec={'kind':'final-score','model':'all16','device':'cuda:0',
            'final_output':str(self.final),'output':str(self.output),
            'daily_csv':str(self.final/worker.SENTINEL_NAME),
            'raw_root':str(self.final.parent/worker.RAW_NAME),
            'allow_download':True,'fit_wall_seconds':1800}
    def signed(self,**changes):
        value={**self.spec,**changes}
        value['worker_spec_sha256']=hashlib.sha256(worker.canonical(value)).hexdigest()
        return value

    def test_only_score_opportunity_preserves_original_raw_root_and_absent_sentinel(self):
        final,raw=worker.validate_opportunity(self.signed(),'final',self.path)
        self.assertEqual(final,self.final)
        self.assertEqual(raw,self.final.parent/'private_raw_E1')
        for changes in ({'kind':'neural-final','model':'FCN-1D'},
                {'kind':'classical','model':'ARIMA'},{'allow_download':False},
                {'output':str(self.final/'score-worker')},
                {'raw_root':str(self.final/'otherraw')},
                {'fit_wall_seconds':600},{'device':'cpu'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                worker.validate_opportunity(self.signed(**changes),'final',self.path)

    def test_existing_daily_source_spec_tampering_and_symlinks_fail_closed(self):
        sentinel=self.final/worker.SENTINEL_NAME
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b'unexpected existing source')
        with self.assertRaises(ValueError):worker.validate_opportunity(self.signed(),'final',self.path)
        sentinel.unlink()
        bad=self.signed();bad['model']='FCN-1D'
        with self.assertRaises(ValueError):worker.validate_opportunity(bad,'final',self.path)
        raw=self.final.parent/worker.RAW_NAME
        raw.symlink_to(self.final,target_is_directory=True)
        with self.assertRaises(ValueError):worker.validate_opportunity(self.signed(),'final',self.path)

    def test_verified_prior_child_route_preserves_decoder_and_forecast_functions(self):
        self.path.write_bytes(worker.canonical(self.signed())+b'\n')
        data=importlib.import_module('e2_reduced_final_data')
        final=importlib.import_module('e2_reduced_final')
        before=(data._download_archive,data.validate_final_freeze,
            data.load_known_e1_evaluation,final.score_final,final._arima_walk)
        arguments=['--arima-root',str(ROOT),'--arima-commit',worker.PRIOR_ARIMA_COMMIT,
            '--mode','final','--engine-root',str(ROOT),'--extension-root',str(ROOT),
            '--extension-commit',worker.RESOURCE_COMMIT,'--worker-spec',str(self.path)]
        calls=[]
        def preserved_child(argv):
            self.assertEqual(argv,arguments)
            self.assertIsNot(data._download_archive,before[0])
            self.assertEqual((data.validate_final_freeze,data.load_known_e1_evaluation,
                final.score_final,final._arima_walk),before[1:])
            calls.append(argv)
            return 0
        class PriorLoader:
            def create_module(self,spec):return None
            def exec_module(self,module):module.main=preserved_child
        original_loader=importlib.util.spec_from_file_location
        def route_loader(name,path,*args,**kwargs):
            if name=='e2_preserved_ARIMA_score_worker':
                return importlib.util.spec_from_loader(name,PriorLoader())
            return original_loader(name,path,*args,**kwargs)
        record={'files_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in worker.FILES}}
        with mock.patch.object(worker,'verify_registration',return_value=record),\
                mock.patch.object(worker.importlib.util,'spec_from_file_location',side_effect=route_loader):
            result=worker.main(['--score-source-root',str(ROOT),'--score-source-commit','a'*40,*arguments])
        self.assertEqual(result,0)
        self.assertEqual(len(calls),1)
        self.assertIs(data._download_archive,before[0])
        admission=json.loads((self.output/'score_source_admission.json').read_bytes())
        self.assertEqual(admission['market_payloads_opened_before_admission'],0)
        self.assertIs(admission['model_refits_requested'],False)
        self.assertEqual(admission['prior_arima_registration_commit'],worker.PRIOR_ARIMA_COMMIT)


if __name__=='__main__':unittest.main()
