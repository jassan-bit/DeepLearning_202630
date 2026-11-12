"""Transport-only checks with an in-memory GitHub; no network or training."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import e2_resilient_github_backup as adapter
from e2_reduced_github_backup import GitHubBackup, BackupPause, RELEASE_TAG


class MemoryGitHub:
    def __init__(self):
        self.releases = {RELEASE_TAG: {'id': 1, 'tag_name': RELEASE_TAG,
            'draft': False, 'prerelease': True}}
        self.assets = {1: {}}
        self.calls = Counter()
        self.upload_ambiguous_once = False
        self.transient_get_count = 0
        self.permanent_get = False

    def __call__(self, client, method, path, payload=None, *, upload=None, accept=None):
        self.calls[(method, path)] += 1
        if path == '/probe':
            if self.permanent_get:
                raise BackupPause('github_request_failed', status=403)
            if self.transient_get_count:
                self.transient_get_count -= 1
                raise BackupPause('github_transport_or_response_failed')
            return {'ok': True}
        if method == 'GET' and path.startswith('/releases/tags/'):
            tag = path.removeprefix('/releases/tags/')
            if tag not in self.releases:
                raise BackupPause('github_request_failed', status=404)
            return self.releases[tag]
        if method == 'POST' and path == '/releases':
            tag = payload['tag_name']
            if tag in self.releases:
                raise BackupPause('github_request_failed', status=422)
            identity = len(self.releases) + 1
            self.releases[tag] = {**payload, 'id': identity}
            self.assets[identity] = {}
            return self.releases[tag]
        if method == 'GET' and '/assets?' in path:
            identity = int(path.split('/')[2])
            return list(self.assets[identity].values())
        if method == 'POST' and '/assets?' in path:
            identity = int(path.split('/')[2])
            name = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)['name'][0]
            if name in self.assets[identity]:
                raise BackupPause('github_request_failed', status=422)
            tag = next(k for k,v in self.releases.items() if v['id'] == identity)
            raw = Path(upload).read_bytes()
            asset = {'id': identity * 1000 + len(self.assets[identity]) + 1,
                'name': name, 'size': len(raw), 'digest':'sha256:' + hashlib.sha256(raw).hexdigest(),
                'browser_download_url':'https://github.com/jassan-bit/DeepLearning_202630/releases/download/' + tag + '/' + name}
            self.assets[identity][name] = asset
            if self.upload_ambiguous_once:
                self.upload_ambiguous_once = False
                raise BackupPause('github_transport_or_response_failed')
            return asset
        raise AssertionError((method, path))


class ResilientBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.network = MemoryGitHub()
        def request(client, *a, **k):
            return self.network(client, *a, **k)
        self.request = patch.object(GitHubBackup, '_request', new=request)
        self.request.start(); self.addCleanup(self.request.stop)
        self.waits = []
        self.client = adapter.ResilientGitHubBackup('synthetic-token', retry_wait=self.waits.append)

    def file(self, value=b'checkpoint'):
        path = self.root / 'checkpoint.pt'; path.write_bytes(value)
        return path, hashlib.sha256(value).hexdigest()

    def test_legacy_asset_retains_exact_tag_url_and_identity(self):
        path, digest = self.file()
        legacy = {'id': 95, 'name':'already-there.pt', 'size':path.stat().st_size,
            'digest':'sha256:' + digest,
            'browser_download_url':'https://github.com/jassan-bit/DeepLearning_202630/releases/download/' + RELEASE_TAG + '/already-there.pt'}
        self.network.assets[1][legacy['name']] = legacy
        result = self.client._asset(path, legacy['name'], digest)
        self.assertEqual(result, {'asset_id':95,'name':legacy['name'],'sha256':digest,
            'bytes':path.stat().st_size,'url':legacy['browser_download_url'],'release_tag':RELEASE_TAG})
        self.assertEqual(len(self.network.releases), 1)
        self.assertFalse(any(m == 'POST' for m,p in self.network.calls))

    def test_new_assets_partition_by_digest_and_release_catalog_is_cached(self):
        path, digest = self.file()
        first = self.client._asset(path, 'first.pt', digest)
        second = self.client._asset(path, 'second.pt', digest)
        tag = RELEASE_TAG + '-p' + digest[0]
        self.assertEqual(first['release_tag'], tag)
        self.assertEqual(second['release_tag'], tag)
        bucket_id = self.network.releases[tag]['id']
        self.assertEqual(self.network.calls[('GET', '/releases/1/assets?per_page=100&page=1')], 1)
        self.assertEqual(self.network.calls[('GET', '/releases/%s/assets?per_page=100&page=1' % bucket_id)], 1)
        self.assertEqual(len(self.network.assets[1]), 0)
        self.assertEqual(len(self.network.assets[bucket_id]), 2)

    def test_ambiguous_upload_reconciles_existing_digest_without_duplicate_post(self):
        path, digest = self.file()
        self.network.upload_ambiguous_once = True
        result = self.client._asset(path, 'one-fit-one-asset.pt', digest)
        self.assertEqual(result['sha256'], digest)
        uploads = [n for (m,p),n in self.network.calls.items() if m == 'POST' and '/assets?' in p]
        self.assertEqual(sum(uploads), 1)
        self.assertEqual(sum(map(len, self.network.assets.values())), 1)

    def test_get_transients_are_bounded_but_permission_failures_are_not_retried(self):
        self.network.transient_get_count = 2
        self.assertEqual(self.client._request('GET','/probe'), {'ok':True})
        self.assertEqual(self.waits,[1,2])
        self.network.permanent_get = True
        previous = self.network.calls[('GET','/probe')]
        with self.assertRaises(BackupPause): self.client._request('GET','/probe')
        self.assertEqual(self.network.calls[('GET','/probe')],previous+1)

    def test_unknown_mutation_is_never_retried_and_wrong_asset_digest_is_rejected(self):
        self.network.transient_get_count = 1
        with self.assertRaises(BackupPause): self.client._request('PATCH','/probe',{'sha':'immutable'})
        self.assertEqual(self.network.calls[('PATCH','/probe')],1)
        path,digest = self.file()
        self.network.assets[1]['changed.pt'] = {'id':95,'name':'changed.pt','size':path.stat().st_size,
            'digest':'sha256:' + '0'*64,'browser_download_url':'https://github.com/file'}
        with self.assertRaisesRegex(BackupPause,'sha256_mismatch'):
            self.client._asset(path,'changed.pt',digest)
        self.assertEqual(len(self.network.releases),1)

    def test_ambiguous_ref_update_reads_intended_sha_without_second_patch(self):
        calls=[]; intended='a'*40
        def request(client,method,path,payload=None,**kwargs):
            calls.append((method,path,payload))
            if method=='PATCH':
                raise BackupPause('github_transport_or_response_failed')
            self.assertEqual(path,'/git/ref/heads/' + adapter.BRANCH)
            return {'object':{'sha':intended}}
        with patch.object(GitHubBackup,'_request',new=request):
            result=self.client._request('PATCH','/git/refs/heads/'+adapter.BRANCH,
                {'sha':intended,'force':False})
        self.assertEqual(result['object']['sha'],intended)
        self.assertEqual([m for m,p,v in calls],['PATCH','GET'])
        self.assertEqual(self.waits,[])

    def test_identical_git_object_creation_retries_but_arbitrary_post_does_not(self):
        calls=[]; payload={'encoding':'base64','content':'c2FtZSBieXRlcw=='}
        def request(client,method,path,body=None,**kwargs):
            calls.append((method,path,body))
            if path=='/git/blobs' and sum(p==path for m,p,v in calls)==2:
                return {'sha':'b'*40}
            raise BackupPause('github_transport_or_response_failed')
        with patch.object(GitHubBackup,'_request',new=request):
            result=self.client._request('POST','/git/blobs',payload)
            with self.assertRaises(BackupPause):
                self.client._request('POST','/issues',{'title':'not-replayed'})
        self.assertEqual(result['sha'],'b'*40)
        self.assertEqual(calls[:2],[('POST','/git/blobs',payload)]*2)
        self.assertEqual(sum(p=='/issues' for m,p,v in calls),1)


if __name__ == '__main__': unittest.main()
