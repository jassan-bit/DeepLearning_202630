"""Integrity/scope guard fixtures; no market/models/network execution."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('capsule',Path(__file__).with_name('integral_capsule.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Guards(unittest.TestCase):
    def test_paths_reject_traversal(self):
        for path in ['../escape','/absolute','a/../b','a\\b','a//b']:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):m.safe(path)
    def test_pinned_bytes_reject_corruption(self):
        raw=b'public fixture';row={'bytes':len(raw),'sha256':m.digest(raw),'git_blob_sha':m.git_digest(raw)}
        m.check(raw,row)
        with self.assertRaises(ValueError):m.check(raw+b'x',row)
    def test_redirect_origin_allowlist(self):
        for url in ['http://github.com/a','https://evil.example/a','https://user:pass@github.com/a']:
            with self.assertRaises(ValueError):m.allowed(url)
    def test_exclude_raw_input_and_unreviewed_weight(self):
        for name,raw in [('data/raw.csv',b'x'),('experiments/metrics.csv',b'open,high,low,close,volume\n1,2,0,1,3\n'),('src/model.pt',b'fake'),('src/state.json',b'{"windows":[[1,2,3,4,5]]}')]:
            self.assertFalse(m.source_permission(name,raw)[0],name)
    def test_notebook_keeps_source_removes_output(self):
        raw=json.dumps({'cells':[{'cell_type':'code','source':['print(1)'],'outputs':[{'data':{'text/plain':'rawdata'}}],'execution_count':1}]}).encode()
        keep,reason,stored=m.source_permission('notebooks/a.ipynb',raw);self.assertTrue(keep)
        cell=json.loads(stored)['cells'][0];self.assertEqual(cell['source'],['print(1)']);self.assertNotIn('outputs',cell)
    def test_budget_fails_closed(self):
        b=m.Budget(5);b.add(5)
        with self.assertRaises(ValueError):b.add(1)
    def test_archive_rebuild_and_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);payload=p/'payload';payload.mkdir();(payload/'a.txt').write_bytes(b'fixture\n')
            a=p/'a.tar.gz';b=p/'b.tar.gz';manifest=m.build_capsule(payload,a)
            manifest2=m.build_capsule(payload,b)
            self.assertEqual(manifest,manifest2);self.assertEqual(a.read_bytes(),b.read_bytes())
            self.assertTrue(m.verify_extraction(a,manifest,p)['all_extracted_hashes_verified'])
            bad=dict(manifest);bad['files']=[dict(manifest['files'][0],sha256='0'*64)]
            with self.assertRaises(ValueError):m.verify_extraction(a,bad,p)
    def test_historical_payload_not_public_by_default(self):
        for path in ['admission/datasets/D01.csv','inputs/daily_market_evaluation.csv','jobs/a/checkpoint.pt','panels/D01/A.csv']:
            self.assertFalse(m.evidence_permission({'group':'audit796','logical_path':path})[0])

if __name__=='__main__':unittest.main()
