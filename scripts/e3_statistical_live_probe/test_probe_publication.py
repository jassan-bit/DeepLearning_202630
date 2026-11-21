"""Pure metadata/transport fixtures. These tests never perform network calls."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlsplit

import probe_publication as p


class ProbeGuards(unittest.TestCase):
    def setUp(self):
        self.expected=json.loads(Path(__file__).with_name('expected_publication.json').read_text())
        self.tmp=tempfile.TemporaryDirectory();self.report=Path(self.tmp.name)/'report.json'
    def tearDown(self):self.tmp.cleanup()
    def good(self,url,is_manifest,timeout):
        path=urlsplit(url).path.removeprefix('/DeepLearning_202630/')
        out={'url':url,'http_status':200,'final_url':url,'headers':{},
             'body_sha256':self.expected['page_sha256'].get(path,'a'*64)}
        if is_manifest:out['manifest_object']=copy.deepcopy(self.expected['manifest'])
        return out
    def run_probe(self,request=None,attempts=3,sleep=lambda _:None):
        with contextlib.redirect_stdout(io.StringIO()):
            return p.probe(self.expected,self.report,attempts=attempts,timeout=10,
                           interval=5,request=request or self.good,sleep=sleep)
    def test_all_five_responses_required_and_exact_source(self):
        out=self.run_probe();self.assertTrue(out['web_deployment_verified'])
        self.assertEqual(out['verified_source_commit'],p.SOURCE)
        self.assertEqual(len(out['attempts'][0]['observations']),5)
    def test_old_served_manifest_is_not_success(self):
        def old(url,is_manifest,timeout):
            r=self.good(url,is_manifest,timeout)
            if is_manifest:r['manifest_object']['source_commit']='d6cf7f1af6800ec1ac85a26347bdde52380e362f'
            return r
        out=self.run_probe(old);self.assertFalse(out['web_deployment_verified'])
        self.assertEqual(len(out['attempts']),3)
        self.assertNotIn('verified_source_commit',out)
    def test_every_individual_page_hash_mismatch_blocks_success(self):
        for bad in p.PATHS:
            with self.subTest(path=bad):
                def mismatch(url,is_manifest,timeout):
                    r=self.good(url,is_manifest,timeout)
                    if urlsplit(url).path.endswith('/'+bad):r['body_sha256']='0'*64
                    return r
                self.assertFalse(self.run_probe(mismatch,attempts=1)['web_deployment_verified'])
    def test_exhausted_errors_have_no_publication_claim(self):
        sleeps=[]
        out=self.run_probe(lambda url,*args:{'url':url,'error_type':'timeout','error':'fixture'},sleep=sleeps.append)
        self.assertFalse(out['web_deployment_verified']);self.assertEqual(len(out['attempts']),3)
        self.assertEqual(sleeps,[5,5]);self.assertTrue(self.report.is_file())
    def test_fixed_source_and_fresh_nonce_not_workflow_HEAD(self):
        def wrong(url,is_manifest,timeout):
            r=self.good(url,is_manifest,timeout)
            if not is_manifest:r['body_sha256']='0'*64
            self.assertIn('source_check='+p.SOURCE,url)
            return r
        with mock.patch.object(p.time,'time_ns',return_value=111):out=self.run_probe(wrong)
        self.assertEqual([a['fresh_nonce'] for a in out['attempts']],['111-1','111-2','111-3'])
    def test_wrong_expected_source_rejected_before_request(self):
        self.expected['source_commit']='f'*40
        with self.assertRaises(ValueError):self.run_probe()
        self.assertFalse(self.report.exists())
    def test_probe_budget_cannot_expand(self):
        for attempts,timeout,interval in [(4,10,5),(3,11,5),(3,10,6)]:
            with self.subTest(values=(attempts,timeout,interval)):
                with self.assertRaises(ValueError):p.probe(self.expected,self.report,attempts,timeout,interval,request=self.good)
    def test_hard_request_deadline_handles_blocked_worker(self):
        with mock.patch.object(p.subprocess,'run',side_effect=subprocess.TimeoutExpired(['worker'],10)) as call:
            out=p.bounded_request(p.SITE+'books_manifest.json',True,10)
        self.assertEqual(out['error_type'],'hard_request_deadline')
        self.assertEqual(call.call_args.kwargs['timeout'],10)
    def test_worker_uses_no_auth_or_cookie_headers(self):
        class Response(io.BytesIO):
            status=200;headers={}
            def geturl(self):return p.SITE+'intro.html'
        def fake(request,timeout):
            self.assertNotIn('Authorization',request.headers);self.assertNotIn('Cookie',request.headers)
            self.assertEqual(request.get_header('Accept-encoding'),'identity')
            return Response(b'fixture')
        with mock.patch.object(p,'urlopen',side_effect=fake):out=p.fetch_once(p.SITE+'intro.html')
        self.assertEqual(out['body_sha256'],p.sha(b'fixture'));self.assertNotIn('error_type',out)
    def test_worker_rejects_arbitrary_initial_origin_or_path_without_network(self):
        for url in ['http://jassan-bit.github.io/DeepLearning_202630/intro.html',
                    'https://example.com/DeepLearning_202630/intro.html',p.SITE+'private.html']:
            with self.subTest(url=url),mock.patch.object(p,'urlopen') as request:
                out=p.fetch_once(url)
                self.assertEqual(out['error_type'],'ValueError');request.assert_not_called()
    def test_redirect_handler_rejects_cross_origin_and_other_entrypoint(self):
        for final in ['https://example.com/DeepLearning_202630/intro.html',
                      p.SITE+'entregable3/intro.html']:
            with self.subTest(final=final):
                with self.assertRaises(ValueError):
                    p.SamePublicationRedirect().redirect_request(
                        p.Request(p.SITE+'intro.html'),None,302,'fixture',{},final)
    def test_response_target_is_checked_before_reading_body(self):
        class Response(io.BytesIO):
            status=200;headers={}
            def geturl(self):return p.SITE+'entregable3/intro.html'
            def read(self,*args):raise AssertionError('must reject before reading')
        with mock.patch.object(p,'urlopen',return_value=Response(b'')):
            out=p.fetch_once(p.SITE+'intro.html')
        self.assertEqual(out['error_type'],'ValueError')


if __name__=='__main__':unittest.main()
