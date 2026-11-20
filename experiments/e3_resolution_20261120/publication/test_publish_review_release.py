"""Release guard tests with fixtures; no network, GitHub writes or model runs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('release_guard',Path(__file__).with_name('publish_review_release.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReleaseGuards(unittest.TestCase):
    def test_new_public_asset_visibility_can_retry_without_writes(self):
        expected={'http':200,'bytes':1,'sha256':'fixture'}
        err=HTTPError('https://github.com/fixture',404,'not propagated',None,None)
        with patch.object(m,'_public_digest_once',side_effect=[err,expected]) as get,patch.object(m.time,'sleep') as pause:
            self.assertEqual(m.public_digest('https://github.com/fixture',{}),expected)
            self.assertEqual(get.call_count,2);pause.assert_called_once()
    def test_public_download_origin_restrictions(self):
        for url in ['http://github.com/a','https://evil.example/a','https://user:pass@github.com/a']:
            with self.subTest(url=url),self.assertRaises(ValueError):m.allowed(url)
    def test_download_bytes_and_sha_checked(self):
        class Response:
            status=200;url='https://release-assets.githubusercontent.com/fixture'
            def __init__(self):self.parts=[b'fixture',b'']
            def read(self,n):return self.parts.pop(0)
            def __enter__(self):return self
            def __exit__(self,*args):return None
        class Opener:
            def open(self,*args,**kwargs):return Response()
        with patch.object(m,'build_opener',return_value=Opener()):
            result=m.public_digest('https://github.com/fixture',{'bytes':7,'sha256':hashlib.sha256(b'fixture').hexdigest()})
            self.assertEqual(result['http'],200)
            for wanted in [{'bytes':6,'sha256':'0'*64},{'bytes':7,'sha256':'0'*64}]:
                with self.assertRaises(ValueError):m.public_digest('https://github.com/fixture',wanted)
    def test_invalid_inputs_never_call_GitHub(self):
        invalid=[{'status':'not_complete'},
                 {'status':'scoped_archive_consolidated_with15_exact_approved_ONNX','source_commit':'wrong'},
                 {'status':'scoped_archive_consolidated_with15_exact_approved_ONNX',
                  'source_commit':'8d03a73d529eb3f9b2e89992b4488cfeb88a99e6','version':'0.3.1-REVIEW',
                  'R3_accredited':False,'full_historical_master_deposited':False,
                  'verification':{'unique_ONNX_graphs_verified':6,'all_member_hashes_verified':True},
                  'C_addon':{'included':True}}]
        for receipt in invalid:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'consolidation_receipt.json').write_text(json.dumps(receipt));report=root/'report.json'
                with patch.dict(os.environ,{'GITHUB_SHA':'a'*40}),patch.object(sys,'argv',['publisher','--directory',tmp,'--report',str(report)]),patch.object(m,'gh',side_effect=AssertionError('No GitHub call is permitted for invalid inputs')) as gh_mock:
                    self.assertEqual(m.main(),2);gh_mock.assert_not_called()
                self.assertEqual(json.loads(report.read_text())['status'],'not_published')

if __name__=='__main__':unittest.main()
