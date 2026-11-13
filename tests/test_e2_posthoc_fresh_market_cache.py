import hashlib,io,json,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_fresh_market_cache as m

class Response(io.BytesIO):
    def __init__(self,data,url):super().__init__(data);self.url=url
    def geturl(self):return self.url

class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.source=self.root/'source';self.output=self.root/'original'
        (self.source/'scripts').mkdir(parents=True);self.loader_file=self.source/'scripts/e2_reduced_final_data.py';self.loader_file.write_bytes(b'original source fixture')
        final=self.output/'final';final.mkdir(parents=True)
        self.weights=final/'original_checkpoint.pt';self.weights.write_bytes(b'original checkpoint bytes')
        self.freeze={'models':81,'checkpoint_sha256':m.sha(self.weights.read_bytes())}
        self.freeze_raw=m.canonical(self.freeze)+b'\n';(final/'final_freeze.json').write_bytes(self.freeze_raw)
        self.records=[{'archive':'BTCUSDT-1m-'+month+'.zip','relative_path':'BTCUSDT/1m/monthly/BTCUSDT-1m-'+month+'.zip',
            'url':'https://data.binance.vision/'+month,'bytes':len(month.encode()),'sha256':m.sha(month.encode())} for month in m.MONTHS]
        self.reservation={'sha256':'b'*64,'archives':self.records}
        frame=self.frame(pd.date_range('2025-01-01','2026-08-31',tz='UTC'))
        self.expected_raw=frame.to_csv().encode();self.cache_sha=m.sha(self.expected_raw)
        self.admission={'freeze_sha256':'a'*64,'reservation_sha256':'b'*64,'daily_market_sha256':self.cache_sha,'completed_at_utc':'original-time-never-replaced'}
        self.admission_raw=m.canonical(self.admission)+b'\n'
        self.admission_path=final/'score-worker-source-recovery-v1/scientific_outputs/data_admission.json'
        self.admission_path.parent.mkdir(parents=True);self.admission_path.write_bytes(self.admission_raw)
        self.cache=final/'private-known-E1-data/known_E1_daily.csv';self.receipt=self.cache.parent/'known_E1_data_receipt.json'
        self.calls=[];self.downloads=[];self.active=0;self.maxactive=0;self.lock=threading.Lock();self.aggregate_months=[]
        self.bad_response=False;self.bad_aggregate=False
        self.loader=SimpleNamespace(validate_final_freeze=self.validate, reserve_known_e1_evaluation=lambda source:self.reservation,
                                    _download_archive=self.download)
    @staticmethod
    def frame(index):
        f=pd.DataFrame([[1.,2.,.5,1.5,10.]]*len(index),columns=['open','high','low','close','volume'],index=index)
        f.index.name='market_day_utc';return f
    def validate(self,freeze,root):
        if freeze['models']!=81 or m.sha((root/'original_checkpoint.pt').read_bytes())!=freeze['checkpoint_sha256']:
            raise ValueError('physical final freeze changed')
        return {'freeze_sha256':'a'*64}
    def opener(self,url,timeout=60):
        with self.lock:self.calls.append(url);self.active+=1;self.maxactive=max(self.maxactive,self.active)
        try:time.sleep(.01);raw=b'corrupt' if self.bad_response else url.rsplit('/',1)[1].encode();return Response(raw,url)
        finally:
            with self.lock:self.active-=1
    def download(self,record,target,opener):
        self.downloads.append(record['archive'])
        if target.exists():return target.read_bytes()
        with opener(record['url'],timeout=60) as response:raw=response.read(record['bytes']+1)
        if len(raw)!=record['bytes'] or m.sha(raw)!=record['sha256']:raise ValueError('download SHA differs')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);return raw
    def aggregate(self,raw,month,expected_sha256):
        if m.sha(raw)!=expected_sha256 or raw.decode()!=month:raise ValueError('aggregation input binding changed')
        self.aggregate_months.append(month);start=pd.Timestamp(month+'-01',tz='UTC');end=start+pd.offsets.MonthBegin()
        f=self.frame(pd.date_range(start,end,inclusive='left'))
        if self.bad_aggregate:f.iloc[-1,4]=900.
        return f,{'synthetic_quality':True}
    def patches(self):
        return (patch.multiple(m,FREEZE_SHA=m.sha(self.freeze_raw),LOADER_SHA=m.sha(self.loader_file.read_bytes()),
                               ADMISSION_SHA=m.sha(self.admission_raw),CACHE_SHA=self.cache_sha,PANDAS_VERSION=pd.__version__),
                patch.object(m.subprocess,'check_output',return_value=m.ENGINE_COMMIT+'\n'),
                patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=0)),
                patch.object(m,'load_original',return_value=(self.loader,self.aggregate)))

class CacheTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.f=Fixture(self.temp.name)
    def run_rebuild(self,parallel=4):
        a,b,c,d=self.f.patches()
        with a,b,c,d:return m.rebuild_original_cache(self.f.source,self.f.output,parallel=parallel,opener=self.f.opener)
    def test_twenty_archives_parallel_but_aggregation_chronological_exact_bytes(self):
        result=self.run_rebuild()
        self.assertEqual(len(self.f.calls),20);self.assertLessEqual(self.f.maxactive,4);self.assertGreater(self.f.maxactive,1)
        self.assertEqual(self.f.aggregate_months,list(m.MONTHS));self.assertEqual(self.f.cache.read_bytes(),self.f.expected_raw)
        self.assertEqual(self.f.receipt.read_bytes(),self.f.admission_raw)
        self.assertFalse((self.f.cache.parent/'current_E2_first_known_E1_access.json').exists())
        self.assertFalse(result['scientific_fit_performed']);self.assertFalse(result['model_inference_performed']);self.assertFalse(result['benchmark_recomputed'])
        self.assertTrue(Path(result['restoration_receipt_path']).exists())
    def test_valid_existing_daily_cache_reused_without_raw_access(self):
        self.f.cache.parent.mkdir(parents=True);self.f.cache.write_bytes(self.f.expected_raw)
        result=self.run_rebuild();self.assertTrue(result['existing_daily_cache_reused']);self.assertEqual(self.f.calls,[])
        self.assertEqual(self.f.receipt.read_bytes(),self.f.admission_raw)
    def test_changed_existing_daily_or_receipt_preserved_and_no_downloads(self):
        self.f.cache.parent.mkdir(parents=True);self.f.cache.write_bytes(b'different daily bytes')
        with self.assertRaisesRegex(ValueError,'SHA256'):self.run_rebuild()
        self.assertEqual(self.f.cache.read_bytes(),b'different daily bytes');self.assertEqual(self.f.calls,[])
        self.f.cache.unlink();self.f.receipt.write_bytes(b'different original receipt')
        with self.assertRaisesRegex(ValueError,'receipt differs'):self.run_rebuild()
        self.assertEqual(self.f.receipt.read_bytes(),b'different original receipt');self.assertEqual(self.f.calls,[])
    def test_original_physical_freeze_gate_precedes_any_raw_access(self):
        self.f.weights.write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError,'physical final freeze'):self.run_rebuild()
        self.assertEqual(self.f.calls,[]);self.assertFalse(self.f.cache.exists())
    def test_corrupt_download_or_aggregation_never_creates_original_cache(self):
        self.f.bad_response=True
        with self.assertRaisesRegex(ValueError,'download SHA'):self.run_rebuild()
        self.assertFalse(self.f.cache.exists());self.assertFalse(self.f.receipt.exists())
        self.f.bad_response=False;self.f.bad_aggregate=True
        with self.assertRaisesRegex(ValueError,'original SHA256'):self.run_rebuild()
        self.assertFalse(self.f.cache.exists());self.assertFalse(self.f.receipt.exists())
    def test_original_reservation_and_months_checked_before_download(self):
        self.f.reservation['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'reservation'):self.run_rebuild()
        self.assertEqual(self.f.calls,[])
        self.f.reservation['sha256']='b'*64;self.f.records.reverse()
        with self.assertRaisesRegex(ValueError,'chronological'):self.run_rebuild()
        self.assertEqual(self.f.calls,[])
    def test_atomic_writer_refuses_symlinks_and_existing_unequal_bytes(self):
        p=self.f.root/'atomic.csv';p.write_bytes(b'preserve')
        with self.assertRaisesRegex(ValueError,'differs'):m.atomic_exact(p,b'new',m.sha(b'new'))
        self.assertEqual(p.read_bytes(),b'preserve');p.unlink();p.symlink_to(self.f.admission_path)
        with self.assertRaisesRegex(ValueError,'symlink'):m.atomic_exact(p,b'new',m.sha(b'new'))
    def test_parallel_validation_and_wrong_scientific_source_fail_early(self):
        with self.assertRaises(ValueError):self.run_rebuild(parallel=5)
        a,b,c,d=self.f.patches()
        with a,c,d,patch.object(m.subprocess,'check_output',return_value='different HEAD'):
            with self.assertRaisesRegex(ValueError,'scientific source'):m.rebuild_original_cache(self.f.source,self.f.output,opener=self.f.opener)
        self.assertEqual(self.f.calls,[])
if __name__=='__main__':unittest.main()
