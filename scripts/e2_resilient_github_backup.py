"""Bounded transport recovery and additive checkpoint release partitioning.

The frozen scientific runner is unchanged. Existing release assets retain their
URLs and manifests. New assets use sixteen deterministic SHA256 partitions, so
one failed fit's many durable checkpoints cannot fill the original release.
Reads, content-addressed Git object creation and a named, digest-checked asset
transfer can retry. An ambiguous fast-forward ref update is first reconciled
against the remote SHA. No model fit or budget charge is replayed here.
"""
from __future__ import annotations

from pathlib import Path
import re
import time
import urllib.parse

from e2_reduced_github_backup import (
    BackupPause, GitHubBackup, MAX_ASSET_BYTES, RELEASE_TAG, BRANCH,
)


def _transient(error):
    return (isinstance(error, BackupPause)
        and (error.http_status in (408, 429, 500, 502, 503, 504)
             or error.code in ('github_transport_or_response_failed',
                               'asset_digest_download_verification_failed')))


class ResilientGitHubBackup(GitHubBackup):
    """Compatible backup client with bounded, byte-preserving I/O recovery."""

    def __init__(self, *args, transport_attempts=4, retry_wait=None, **kwargs):
        super().__init__(*args, **kwargs)
        if type(transport_attempts) is not int or not 1 <= transport_attempts <= 6:
            raise ValueError('Use one through six bounded transport attempts')
        self.transport_attempts = transport_attempts
        self._retry_wait = time.sleep if retry_wait is None else retry_wait
        self._release_cache = {}
        self._asset_cache = {}
        self._active_release_tag = None

    def _wait(self, attempt):
        self._retry_wait(min(8, 2 ** attempt))

    def _request(self, method, path, payload=None, *, upload=None,
                 accept='application/vnd.github+json'):
        content_object = (method == 'POST' and upload is None
            and path in ('/git/blobs', '/git/trees', '/git/commits')
            and isinstance(payload, dict))
        safe_ref = (method == 'PATCH' and upload is None
            and path == '/git/refs/heads/' + BRANCH and isinstance(payload, dict)
            and set(payload) == {'sha', 'force'} and payload.get('force') is False
            and isinstance(payload.get('sha'), str)
            and re.fullmatch(r'[0-9a-f]{40}', payload['sha']))
        # Git blobs/trees have content identities. A repeated commit-object
        # creation can at most leave an unreferenced object; it cannot move a
        # branch, alter a completed fit or duplicate a ledger charge.
        retryable = upload is None and (method == 'GET' or content_object or safe_ref)
        attempts = self.transport_attempts if retryable else 1
        for attempt in range(attempts):
            try:
                return super()._request(method, path, payload, upload=upload, accept=accept)
            except BackupPause as error:
                if not _transient(error):
                    raise
                if safe_ref:
                    # A response can be lost after GitHub moved the ref. Read
                    # it before another write; preserve the exact intended SHA.
                    observed = self._request('GET', '/git/ref/heads/' + BRANCH)
                    if observed.get('object', {}).get('sha') == payload['sha']:
                        return observed
                    # Identical force=False updates cannot roll back a remote
                    # advance. GitHub409/422 is left to the frozen CAS loop.
                if attempt + 1 == attempts:
                    raise
                self._wait(attempt)

    @staticmethod
    def _validate_release(result, tag):
        if (result.get('tag_name') != tag or result.get('draft') is not False
                or result.get('prerelease') is not True):
            raise BackupPause('release_identity_or_prerelease_mismatch')

    def _release_for_tag(self, tag):
        if tag in self._release_cache:
            return self._release_cache[tag]
        path = '/releases/tags/' + tag
        for attempt in range(self.transport_attempts):
            try:
                result = self._request('GET', path)
            except BackupPause as error:
                if error.http_status != 404:
                    raise
                try:
                    result = self._request('POST', '/releases', {
                        'tag_name': tag, 'target_commitish': BRANCH,
                        'name': 'E2 reduced immutable artifacts ' + tag,
                        'body': ('Additive SHA256-partitioned E2 backups. '
                            'Existing artifacts and scientific attempts are never replaced.'),
                        'draft': False, 'prerelease': True})
                except BackupPause as uncertain:
                    # GitHub tag uniqueness makes a same-tag creation safe to
                    # reconcile. First read whether the ambiguous POST landed.
                    if uncertain.http_status != 422 and not _transient(uncertain):
                        raise
                    try:
                        result = self._request('GET', path)
                    except BackupPause as missing:
                        if missing.http_status != 404 or attempt + 1 == self.transport_attempts:
                            raise
                        self._wait(attempt)
                        continue
            self._validate_release(result, tag)
            self._release_cache[tag] = result
            return result
        raise BackupPause('release_creation_reconciliation_limit')

    def _select_release(self, tag):
        release = self._release_for_tag(tag)
        self._release = release
        self._active_release_tag = tag
        if tag not in self._asset_cache:
            self._refresh_assets()
        self._assets = self._asset_cache[tag]
        return release

    def _ensure_release(self):
        # Frozen preflight/receipt compatibility retains the original release
        # as the transport anchor; each asset row carries its actual tag/URL.
        return self._select_release(RELEASE_TAG)

    def _refresh_assets(self):
        if self._release is None or self._active_release_tag is None:
            raise BackupPause('select_release_before_listing_assets')
        super()._refresh_assets()
        self._asset_cache[self._active_release_tag] = self._assets

    def _verify_asset(self, asset, expected_sha, expected_size):
        for attempt in range(self.transport_attempts):
            try:
                return super()._verify_asset(asset, expected_sha, expected_size)
            except BackupPause as error:
                if attempt + 1 == self.transport_attempts or not _transient(error):
                    raise
                self._wait(attempt)

    def _asset(self, path, name, expected_sha):
        path = Path(path)
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
            raise BackupPause('invalid_partition_asset_digest')
        size = path.stat().st_size
        if size > MAX_ASSET_BYTES:
            raise BackupPause('checkpoint_exceeds_GitHub_asset_limit')
        # Legacy names are resolved first, preserving old receipt manifests
        # byte-for-byte when an already backed-up result is visited again.
        self._select_release(RELEASE_TAG)
        asset = self._assets.get(name)
        tag = RELEASE_TAG
        if asset is None:
            tag = RELEASE_TAG + '-p' + expected_sha[0]
            self._select_release(tag)
            asset = self._assets.get(name)
        if asset is None:
            endpoint = '/releases/%s/assets?name=%s' % (
                self._release['id'], urllib.parse.quote(name, safe=''))
            for attempt in range(self.transport_attempts):
                try:
                    asset = self._request('POST', endpoint, upload=path)
                    self._assets[name] = asset
                    break
                except BackupPause as error:
                    if error.http_status != 422 and not _transient(error):
                        raise
                    # An upload timeout may follow a completed server upload.
                    # Refresh by its exact unique name before another POST.
                    self._refresh_assets()
                    asset = self._assets.get(name)
                    if asset is not None:
                        break
                    if error.http_status == 422:
                        raise BackupPause('asset_name_race_not_reconciled') from None
                    if attempt + 1 == self.transport_attempts:
                        raise
                    self._wait(attempt)
        self._verify_asset(asset, expected_sha, size)
        return {'asset_id': asset['id'], 'name': name, 'sha256': expected_sha,
            'bytes': size, 'url': asset['browser_download_url'], 'release_tag': tag}
