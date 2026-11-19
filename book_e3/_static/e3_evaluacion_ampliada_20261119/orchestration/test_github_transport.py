"""Offline transport contracts: fake HTTP and clock, no training or real waits."""
import base64
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from email.utils import formatdate
import hashlib
from http.client import IncompleteRead
import importlib.util
import io
import json
from pathlib import Path
import threading
import types
import unittest
from urllib.error import HTTPError, URLError

spec=importlib.util.spec_from_file_location('extended_transport_tested',Path(__file__).with_name('github_transport.py'))
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
ROOT='/repos/jassan-bit/DeepLearning_202630'
TOKEN='github_pat_PRIVATE_TOKEN_123456789'


class StorageError(RuntimeError):pass


def git_sha(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def legacy():
    def no_legacy_calls(*args,**kwargs):raise AssertionError('Legacy network bypassed transport')
    return types.SimpleNamespace(REPOSITORY='jassan-bit/DeepLearning_202630',
        StorageError=StorageError,StorageConflict=StorageError,
        _canonical=lambda p:json.dumps(p,allow_nan=False).encode(),
        _git_sha=git_sha,_token=lambda token:token,_repo=lambda value:value,
        _request=no_legacy_calls,validate_credentials=no_legacy_calls,
        _safe_relative=lambda p:True)


class Clock:
    def __init__(self):self.now=100.;self.sleeps=[];self.events=[]
    def time(self):return self.now
    def sleep(self,value):self.sleeps.append(value);self.now+=value
    def coordinator(self):return T._Coordinator(clock=self.time,sleep=self.sleep,
        wall_time=lambda:1700000000.,emit=self.events.append)


class Response:
    def __init__(self,value,raw=False):self.data=value if raw else json.dumps(value).encode();self.read_sizes=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,size):self.read_sizes.append(size);return self.data[:size]


class Opener:
    def __init__(self,clock,actions):self.clock=clock;self.actions=list(actions);self.calls=[]
    def open(self,request,timeout):
        self.calls.append({'method':request.method,'url':request.full_url,
            'data':request.data,'time':self.clock.now,'headers':dict(request.header_items())})
        action=self.actions.pop(0)
        if isinstance(action,Exception):raise action
        return action


def http(status,message='',retry=None,remaining=None):
    headers=Message()
    if retry is not None:headers['Retry-After']=str(retry)
    if remaining is not None:headers['X-RateLimit-Remaining']=str(remaining)
    return HTTPError('https://api.github.com/private?token='+TOKEN,status,'secret '+TOKEN,
        headers,io.BytesIO(json.dumps({'message':message}).encode()))


def blob(raw,**changes):
    value={'sha':git_sha(raw),'encoding':'base64','size':len(raw),
           'content':base64.encodebytes(raw).decode()}
    value.update(changes);return value


def payload(raw):return {'encoding':'base64','content':base64.b64encode(raw).decode()}


class TransportTests(unittest.TestCase):
    def api(self,actions):
        self.clock=Clock();self.opener=Opener(self.clock,actions)
        return T.wrap(legacy(),_opener=self.opener,_coordinator=self.clock.coordinator())

    def test_proxy_helpers_and_idempotence(self):
        api=self.api([])
        self.assertIs(T.wrap(api),api);self.assertIs(api.StorageError,StorageError)
        self.assertEqual(api._git_sha(b'a'),git_sha(b'a'));self.assertTrue(api._safe_relative('x'))

    def test_credentials_use_new_transport(self):
        api_repo=legacy().REPOSITORY
        api=self.api([Response({'login':'author'}),Response({'full_name':api_repo,'permissions':{'push':True}})])
        result=api.validate_credentials(TOKEN)
        self.assertEqual(result['repository'],api_repo)
        self.assertFalse(result['token_contents_write_verified'])
        self.assertEqual([c['method'] for c in self.opener.calls],['GET','GET'])

    def test_credentials_reject_no_push(self):
        api=self.api([Response({'login':'author'}),Response({'full_name':legacy().REPOSITORY,'permissions':{'push':False}})])
        with self.assertRaisesRegex(StorageError,'repository_push_access_required'):api.validate_credentials(TOKEN)

    def test_credentials_malformed_metadata_has_safe_diagnostic(self):
        for malformed in ({'full_name':None,'permissions':{}},
                          {'full_name':legacy().REPOSITORY,'permissions':None},
                          {'full_name':[],'permissions':{'push':True}}):
            api=self.api([Response({'login':'author'}),Response(malformed)])
            with self.assertRaisesRegex(StorageError,'repository_push_access_required'):
                api.validate_credentials(TOKEN)

    def test_404_optional_only(self):
        api=self.api([http(404),http(404)])
        self.assertIsNone(api._request('GET',ROOT+'/git/ref/heads/x',TOKEN,allow_missing=True))
        with self.assertRaisesRegex(StorageError,'status=404'):api._request('GET','/user',TOKEN)

    def test_mutations_paced_across_proxies(self):
        api=self.api([Response({'ok':1}),Response({'ok':2})])
        other=T.wrap(legacy(),_opener=self.opener,_coordinator=api._coordinator)
        api._request('POST',ROOT+'/git/trees',TOKEN,{})
        other._request('POST',ROOT+'/git/commits',TOKEN,{})
        self.assertGreaterEqual(self.opener.calls[1]['time']-self.opener.calls[0]['time'],1.1-1e-12)
        self.assertTrue(all(x<=30 for x in self.clock.sleeps))

    def test_secondary_403_retries_mutation_and_honors_60(self):
        api=self.api([http(403,'You have exceeded a secondary rate limit',60,4546),Response({'sha':'ok'})])
        self.assertEqual(api._request('POST',ROOT+'/git/trees',TOKEN,{})['sha'],'ok')
        self.assertEqual(self.clock.sleeps,[30.,30.])
        self.assertEqual(self.opener.calls[1]['time']-self.opener.calls[0]['time'],60)

    def test_429_retries_with_larger_retry_after(self):
        api=self.api([http(429,'',75),Response({})])
        api._request('POST',ROOT+'/git/tags',TOKEN,{})
        self.assertEqual(self.clock.sleeps,[30.,30.,15.])

    def test_retry_after_http_date(self):
        api=self.api([http(429,retry=formatdate(1700000120,usegmt=True)),Response({})])
        api._request('GET','/user',TOKEN)
        self.assertEqual(sum(self.clock.sleeps),120)

    def test_impractical_retry_after_refuses_early_retry_and_no_sleep(self):
        api=self.api([http(429,retry='1e300')])
        with self.assertRaisesRegex(StorageError,'wait_exceeds_budget_retry_later'):
            api._request('POST',ROOT+'/git/refs',TOKEN,{})
        with self.assertRaisesRegex(StorageError,'wait_exceeds_budget_retry_later'):
            api._request('GET','/user',TOKEN)
        self.assertEqual(self.clock.sleeps,[]);self.assertEqual(len(self.opener.calls),1)

    def test_clock_advance_at_wait_boundary_does_not_sleep_negative(self):
        values=iter([99.9,100.1])
        sleeps=[]
        coordinator=T._Coordinator(clock=lambda:next(values),sleep=sleeps.append,emit=lambda value:None)
        coordinator.next_mutation=100.
        coordinator.wait(True,'github_write')
        self.assertEqual(sleeps,[100.-99.9])

    def test_invalid_retry_after_defaults_at_least_60(self):
        for retry in (None,'nonsense','-5','NaN','inf','0'):
            with self.subTest(retry=retry):
                api=self.api([http(429,retry=retry),Response({})]);api._request('GET','/user',TOKEN)
                self.assertEqual(sum(self.clock.sleeps),60)

    def test_abuse_rejection_is_secondary(self):
        api=self.api([http(403,'abuse detection mechanism',retry=10),Response({})])
        api._request('POST',ROOT+'/git/refs',TOKEN,{})
        self.assertEqual(sum(self.clock.sleeps),60)

    def test_auth_and_permissions_never_retry(self):
        for status in (401,403):
            with self.subTest(status=status):
                api=self.api([http(status,'Resource not accessible by personal access token',retry=60)])
                with self.assertRaisesRegex(StorageError,'auth_or_contents_write_required'):
                    api._request('POST',ROOT+'/git/refs',TOKEN,{})
                self.assertEqual(len(self.opener.calls),1);self.assertEqual(self.clock.sleeps,[])

    def test_primary_403_is_not_automatic_mutation_retry(self):
        api=self.api([http(403,'API rate limit exceeded',remaining=0)])
        with self.assertRaises(StorageError):api._request('POST',ROOT+'/git/refs',TOKEN,{})
        self.assertEqual(len(self.opener.calls),1)

    def test_primary_limit_has_distinct_safe_reset_diagnostic_without_auth_or_replay(self):
        api=self.api([http(403,'API rate limit exceeded '+TOKEN,remaining=0)])
        with self.assertRaisesRegex(StorageError,'primary_rate_limit_retry_after_reset') as caught:
            api._request('POST',ROOT+'/git/refs',TOKEN,{'ref':'refs/heads/results'})
        self.assertNotIn('auth_or_contents',str(caught.exception))
        self.assertNotIn(TOKEN,str(caught.exception))
        self.assertIn('status=403',str(caught.exception))
        self.assertEqual(len(self.opener.calls),1);self.assertEqual(self.clock.sleeps,[])

    def test_retry_budget_bounded_and_cooldown_global_after_failure(self):
        api=self.api([http(429) for _ in range(4)]+[Response({})])
        with self.assertRaisesRegex(StorageError,'retry_budget_exhausted.*attempts=4'):
            api._request('POST',ROOT+'/git/refs',TOKEN,{})
        self.assertEqual(len(self.opener.calls),4);self.assertEqual(sum(self.clock.sleeps),180)
        other=T.wrap(legacy(),_opener=self.opener,_coordinator=api._coordinator)
        other._request('GET','/user',TOKEN)
        self.assertEqual(sum(self.clock.sleeps),240)

    def test_uncertain_mutations_and_5xx_not_retried(self):
        for failure in (URLError(TOKEN),TimeoutError(TOKEN),IncompleteRead(TOKEN.encode()),http(500,TOKEN),Response(b'{broken',raw=True)):
            with self.subTest(kind=type(failure).__name__):
                api=self.api([failure])
                with self.assertRaises(StorageError) as caught:api._request('POST',ROOT+'/git/refs',TOKEN,{})
                self.assertEqual(len(self.opener.calls),1)
                self.assertNotIn(TOKEN,str(caught.exception))

    def test_existing_blob_verified_without_post(self):
        raw=b'known evidence\n'*10;api=self.api([Response(blob(raw))])
        self.assertEqual(api._request('POST',ROOT+'/git/blobs',TOKEN,payload(raw)),{'sha':git_sha(raw)})
        self.assertEqual([c['method'] for c in self.opener.calls],['GET'])

    def test_missing_blob_posts_and_checks_digest(self):
        raw=b'new evidence';api=self.api([http(404),Response({'sha':git_sha(raw)})])
        self.assertEqual(api._request('POST',ROOT+'/git/blobs',TOKEN,payload(raw))['sha'],git_sha(raw))
        self.assertEqual([c['method'] for c in self.opener.calls],['GET','POST'])

    def test_existing_blob_corruption_never_posts(self):
        raw=b'evidence'
        for change in ({'sha':'f'*40},{'size':9},{'size':True},{'encoding':'utf8'},
                       {'content':'%%%'},{'content':base64.b64encode(b'other').decode()}):
            with self.subTest(change=change):
                api=self.api([Response(blob(raw,**change))])
                with self.assertRaisesRegex(StorageError,'existing_blob_integrity_mismatch'):
                    api._request('POST',ROOT+'/git/blobs',TOKEN,payload(raw))
                self.assertEqual(len(self.opener.calls),1)

    def test_existing_blob_wrong_shape_never_posts(self):
        for value in ([],{},None):
            with self.subTest(value=value):
                api=self.api([Response(value)])
                with self.assertRaises(StorageError):api._request('POST',ROOT+'/git/blobs',TOKEN,payload(b'a'))
                self.assertEqual(len(self.opener.calls),1)

    def test_probe_permissions_never_fall_back_to_post(self):
        api=self.api([http(403,'forbidden')])
        with self.assertRaises(StorageError):api._request('POST',ROOT+'/git/blobs',TOKEN,payload(b'a'))
        self.assertEqual(len(self.opener.calls),1)

    def test_blob_probe_limit_boundary(self):
        raw=b'x'*T.MAX_PROBE;api=self.api([Response(blob(raw))])
        api._request('POST',ROOT+'/git/blobs',TOKEN,payload(raw))
        self.assertEqual(self.opener.calls[0]['method'],'GET')
        raw+=b'x';api=self.api([Response({'sha':git_sha(raw)})])
        api._request('POST',ROOT+'/git/blobs',TOKEN,payload(raw))
        self.assertEqual([c['method'] for c in self.opener.calls],['POST'])

    def test_invalid_blob_input_rejected(self):
        for value in ('%%%',None,'é'):
            api=self.api([])
            with self.assertRaises(StorageError):api._request('POST',ROOT+'/git/blobs',TOKEN,{'encoding':'base64','content':value})
            self.assertEqual(self.opener.calls,[])

    def test_uploaded_blob_wrong_digest_fails(self):
        api=self.api([http(404),Response({'sha':'0'*40})])
        with self.assertRaisesRegex(StorageError,'uploaded_blob_integrity_mismatch'):
            api._request('POST',ROOT+'/git/blobs',TOKEN,payload(b'a'))

    def test_oversized_response_bounded_read_and_no_retry(self):
        response=Response(b'x'*(T.MAX_RESPONSE+1),raw=True);api=self.api([response])
        with self.assertRaisesRegex(StorageError,'response_too_large'):api._request('POST',ROOT+'/git/refs',TOKEN,{})
        self.assertEqual(response.read_sizes,[T.MAX_RESPONSE+1]);self.assertEqual(len(self.opener.calls),1)

    def test_malformed_json_and_scalar_rejected(self):
        for raw in (b'{broken',b'null',b'true',b'12',b'"text"',b'\xff'):
            api=self.api([Response(raw,raw=True)])
            with self.assertRaises(StorageError):api._request('GET','/user',TOKEN)
        api=self.api([Response([])]);self.assertEqual(api._request('GET',ROOT+'/git/matching-refs/heads/',TOKEN),[])

    def test_safe_diagnostics_never_disclose_url_body_or_token(self):
        api=self.api([http(403,'You have exceeded a secondary rate limit '+TOKEN,60),http(403,TOKEN)])
        with self.assertRaises(StorageError) as caught:api._request('POST',ROOT+'/git/refs?private=hidden',TOKEN,{'private':'payload-secret'})
        diagnostic=str(caught.exception)+'\n'+'\n'.join(self.clock.events)
        for private in (TOKEN,'payload-secret','private=hidden','https://','You have exceeded'):
            self.assertNotIn(private,diagnostic)
        self.assertIn('git_refs_write',diagnostic);self.assertIn('status=403',diagnostic)

    def test_path_redirects_and_credentials_rejected_without_http(self):
        for path in ('//evil.test/path','/user#secret','/user\nAuthorization:x','/'+TOKEN,'/x\\y'):
            api=self.api([])
            with self.assertRaises(StorageError):api._request('GET',path,TOKEN)
            self.assertEqual(self.opener.calls,[])
        api=self.api([])
        with self.assertRaisesRegex(StorageError,'credential_in_payload_refused'):
            api._request('POST',ROOT+'/git/refs',TOKEN,{'secret':TOKEN})

    def test_raw_blob_is_verified_and_uses_shared_cooldown(self):
        raw=b'checkpoint bytes';api=self.api([http(429),Response(raw,raw=True)])
        self.assertEqual(api.raw_blob(git_sha(raw),len(raw),TOKEN),raw)
        self.assertEqual(sum(self.clock.sleeps),60)
        self.assertIn('raw+json',self.opener.calls[-1]['headers']['Accept'])

    def test_continuation_blob_delegates_to_serialized_raw_transport(self):
        spec=importlib.util.spec_from_file_location('transport_continuation_contract',
            Path(__file__).with_name('continuation.py'))
        continuation=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuation)
        raw=b'opaque checkpoint, never unpickled';api=self.api([Response(raw,raw=True)])
        self.assertEqual(continuation._blob(api,TOKEN,git_sha(raw),len(raw)),raw)
        self.assertEqual(len(self.opener.calls),1)
        self.assertEqual(self.opener.calls[0]['method'],'GET')
        self.assertIn('raw+json',self.opener.calls[0]['headers']['Accept'])

    def test_raw_blob_wrong_bytes_size_or_identity(self):
        for raw in (b'bad',b'toolong'):
            api=self.api([Response(raw,raw=True)])
            with self.assertRaises(StorageError):api.raw_blob(git_sha(b'good'),4,TOKEN)
        for sha,size in (('no',1),('a'*40,-1),('a'*40,True),('a'*40,T.MAX_RAW_BLOB+1)):
            api=self.api([])
            with self.assertRaises(StorageError):api.raw_blob(sha,size,TOKEN)
            self.assertEqual(self.opener.calls,[])

    def test_parallel_calls_serialized_including_full_response_read(self):
        entered=threading.Event();release=threading.Event();second_lock=threading.Event()
        api=self.api([Response({}),Response({})]);real_lock=api._coordinator.lock
        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name.endswith('_1'):second_lock.set()
                real_lock.acquire()
            def __exit__(self,*args):real_lock.release()
        api._coordinator.lock=ObservedLock()
        class BlockingResponse(Response):
            def read(self,size):
                entered.set()
                if not release.wait(3):raise AssertionError('Synchronization failed')
                return super().read(size)
        self.opener.actions[0]=BlockingResponse({})
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='github_transport_test') as pool:
            first=pool.submit(api._request,'POST',ROOT+'/git/trees',TOKEN,{})
            self.assertTrue(entered.wait(3))
            second=pool.submit(api._request,'GET','/user',TOKEN)
            try:
                self.assertTrue(second_lock.wait(3));self.assertEqual(len(self.opener.calls),1)
            finally:release.set()
            self.assertEqual(first.result(),{});self.assertEqual(second.result(),{})
        self.assertEqual(len(self.opener.calls),2)


if __name__=='__main__':unittest.main()
