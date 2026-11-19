"""Serialized GitHub transport for recoverable audit persistence.

Historical API/store modules stay unchanged. Only an explicitly rejected rate
limit can repeat a mutation; an uncertain network write must be reconciled by
the existing immutable store. Credentials and server bodies never enter logs.
"""
from __future__ import annotations

import base64
import binascii
from email.utils import parsedate_to_datetime
from http.client import HTTPException
import json
import math
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_RESPONSE = 16 * 1024**2
MAX_ERROR = 64 * 1024
MAX_PROBE = 4 * 1024**2
MAX_PROBE_RESPONSE = 6 * 1024**2  # JSON/base64 line breaks for <=4 MiB decoded.
MAX_RAW_BLOB = 48 * 1024**2
MAX_ATTEMPTS = 4
MAX_WAIT = 3600.0
MUTATION_INTERVAL = 1.1
MIN_COOLDOWN = 60.0
SLEEP_CHUNK = 30.0


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _Coordinator:
    """One lock/cooldown for every proxy created from this transport module."""
    def __init__(self, *, clock=time.monotonic, sleep=time.sleep, wall_time=time.time,
                 emit=None):
        self.clock, self.sleep, self.wall_time = clock, sleep, wall_time
        self.emit = emit or (lambda value: print(value, flush=True))
        self.lock = threading.RLock()
        self.next_mutation = 0.0
        self.cooldown_until = 0.0

    def event(self, name, operation, **fields):
        # All fields are locally generated numbers/fixed labels, never a URL,
        # payload, token, server error message, exception repr or header value.
        self.emit(json.dumps({'event': name, 'operation': operation, **fields},
                             sort_keys=True, allow_nan=False))

    def wait(self, mutation, operation):
        target = max(self.cooldown_until, self.next_mutation if mutation else 0.0)
        while True:
            remaining = target - self.clock()
            if remaining <= 0:
                break
            chunk = min(SLEEP_CHUNK, remaining)
            self.event('github_transport_wait', operation,
                       remaining_seconds=round(remaining, 3), sleep_seconds=chunk)
            self.sleep(chunk)


_GLOBAL = _Coordinator()


def _operation(method, path):
    # Path content is untrusted; only a fixed resource classification is logged.
    if re.search(r'/git/blobs(?:/[^/?]+)?(?:\?|$)', path):
        return 'git_blob_create' if method == 'POST' else 'git_blob_read'
    for resource in ('trees', 'commits', 'refs', 'tags'):
        if '/git/' + resource in path or '/git/ref/' in path:
            return 'git_' + resource + '_' + ('write' if method != 'GET' else 'read')
    return 'github_' + ('read' if method in {'GET', 'HEAD'} else 'write')


class _Proxy:
    _e3_serial_transport = True

    def __init__(self, legacy, opener, coordinator):
        self._legacy = legacy
        self._opener = opener
        self._coordinator = coordinator

    def __getattr__(self, name):
        return getattr(self._legacy, name)

    def _fail(self, diagnostic, operation, *, status=None, attempt=None):
        fields = ['github_transport_' + diagnostic, 'operation=' + operation]
        if status is not None:
            fields.append('status=' + str(int(status)))
        if attempt is not None:
            fields.append('attempts=' + str(int(attempt)))
        raise self.StorageError(' '.join(fields)) from None

    def _inputs(self, method, path, token, payload):
        token = self._token(token)
        if (not isinstance(method, str) or method not in {'GET','HEAD','POST','PATCH','PUT','DELETE'}
                or not isinstance(path, str) or not path.startswith('/') or path.startswith('//')
                or '\\' in path or any(ord(c) < 32 or ord(c) == 127 for c in path)
                or token in path):
            self._fail('invalid_request', 'github_request')
        parsed = urlsplit('https://api.github.com' + path)
        if parsed.netloc != 'api.github.com' or parsed.fragment:
            self._fail('invalid_request', 'github_request')
        try:
            raw = None if payload is None else self._canonical(payload)
        except (ValueError, TypeError, UnicodeError):
            self._fail('invalid_payload', _operation(method, path))
        if raw is not None and token.encode() in raw:
            self._fail('credential_in_payload_refused', _operation(method, path))
        return token, raw

    def _retry_after(self, headers):
        delay = MIN_COOLDOWN
        value = headers.get('Retry-After') if headers is not None else None
        if value is not None:
            try:
                seconds = float(value)
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                try:
                    seconds = parsedate_to_datetime(value).timestamp() - self._coordinator.wall_time()
                    if not math.isfinite(seconds):
                        raise ValueError
                except (ValueError, TypeError, OverflowError, AttributeError):
                    seconds = MIN_COOLDOWN
            delay = max(delay, seconds)
        return delay

    def _limited(self, error):
        if error.code == 429:
            return True
        if error.code != 403:
            return False
        try:
            raw = error.read(MAX_ERROR + 1)
            if len(raw) > MAX_ERROR:
                return False
            message = json.loads(raw).get('message', '')
        except (ValueError, UnicodeError, TypeError, AttributeError, OSError, HTTPException):
            return False
        return (isinstance(message, str) and any(marker in message.lower() for marker in
                ('secondary rate limit', 'abuse detection', 'abuse rate limit')))

    def _perform(self, method, path, token, data=None, *, allow_missing=False,
                 accept='application/vnd.github+json', limit=MAX_RESPONSE, binary=False):
        operation = _operation(method, path)
        mutation = method not in {'GET','HEAD'}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self._coordinator.cooldown_until - self._coordinator.clock() > MAX_WAIT:
                self._fail('rate_limit_wait_exceeds_budget_retry_later', operation, attempt=attempt)
            self._coordinator.wait(mutation, operation)
            request = Request('https://api.github.com' + path, data=data, method=method,
                headers={'Authorization': 'Bearer ' + token,
                    'Accept': accept, 'X-GitHub-Api-Version': '2022-11-28',
                    'Content-Type': 'application/json', 'User-Agent': 'E3-Extended-Audit-Transport'})
            if mutation:
                self._coordinator.next_mutation = self._coordinator.clock() + MUTATION_INTERVAL
            try:
                with self._opener.open(request, timeout=90) as response:
                    raw = response.read(limit + 1)
                if not isinstance(raw, bytes) or len(raw) > limit:
                    self._fail('response_too_large_or_invalid', operation, attempt=attempt)
                if binary:
                    return raw
                value = json.loads(raw)
                if not isinstance(value, (dict, list)):
                    self._fail('invalid_json_shape', operation, attempt=attempt)
                return value
            except HTTPError as error:
                try:
                    status = error.code
                    if status == 404 and allow_missing:
                        return None
                    limited = self._limited(error)
                    primary = (status == 403 and error.headers is not None
                               and error.headers.get('X-RateLimit-Remaining') == '0')
                    delay = self._retry_after(error.headers) if limited else None
                finally:
                    error.close()
                if limited:
                    self._coordinator.cooldown_until = max(self._coordinator.cooldown_until,
                        self._coordinator.clock() + delay)
                    self._coordinator.event('github_rate_limit_rejected', operation,
                        status=status, attempt=attempt, max_attempts=MAX_ATTEMPTS,
                        cooldown_seconds=delay)
                    if delay > MAX_WAIT:
                        self._fail('rate_limit_wait_exceeds_budget_retry_later', operation,
                                   status=status, attempt=attempt)
                    if attempt == MAX_ATTEMPTS:
                        self._fail('rate_limit_retry_budget_exhausted', operation,
                                   status=status, attempt=attempt)
                    continue
                if primary:
                    self._fail('primary_rate_limit_retry_after_reset', operation,
                               status=status, attempt=attempt)
                self._fail('auth_or_contents_write_required' if status in (401,403)
                           else 'http_error', operation, status=status, attempt=attempt)
            except (URLError, TimeoutError, OSError, HTTPException):
                self._fail('network_outcome_unconfirmed_reconcile_snapshot' if mutation
                           else 'network_read_failed', operation, attempt=attempt)
            except (ValueError, UnicodeError, TypeError):
                self._fail('invalid_json_response', operation, attempt=attempt)

    def _small_blob(self, payload):
        if not isinstance(payload, dict) or payload.get('encoding') != 'base64':
            return None
        content = payload.get('content')
        if not isinstance(content, str):
            self._fail('invalid_blob_payload', 'git_blob_create')
        if len(content) > 4 * ((MAX_PROBE + 2) // 3):
            return None
        try:
            raw = base64.b64decode(content, validate=True)
        except (ValueError, binascii.Error, UnicodeError):
            self._fail('invalid_blob_payload', 'git_blob_create')
        if len(raw) > MAX_PROBE:
            return None
        return raw

    def _request(self, method, path, token, payload=None, *, allow_missing=False):
        token, data = self._inputs(method, path, token, payload)
        with self._coordinator.lock:
            small = self._small_blob(payload) if method == 'POST' and path.endswith('/git/blobs') else None
            expected = self._git_sha(small) if small is not None else None
            if expected is not None:
                found = self._perform('GET', path + '/' + expected, token, allow_missing=True,
                    limit=MAX_PROBE_RESPONSE)
                if found is not None:
                    try:
                        decoded = base64.b64decode(''.join(found['content'].split()), validate=True)
                        valid = (found.get('sha') == expected and found.get('encoding') == 'base64'
                            and type(found.get('size')) is int and found['size'] == len(small)
                            and decoded == small and self._git_sha(decoded) == expected)
                    except (ValueError, KeyError, TypeError, AttributeError, binascii.Error, UnicodeError):
                        valid = False
                    if not valid:
                        self._fail('existing_blob_integrity_mismatch', 'git_blob_read')
                    self._coordinator.event('github_existing_blob_verified', 'git_blob_read', bytes=len(small))
                    return {'sha': expected}
            result = self._perform(method, path, token, data, allow_missing=allow_missing)
            if expected is not None and (not isinstance(result, dict) or result.get('sha') != expected):
                self._fail('uploaded_blob_integrity_mismatch', 'git_blob_create')
            return result

    def validate_credentials(self, token, *, repository=None):
        token = self._token(token)
        repository = self._repo(self.REPOSITORY if repository is None else repository)
        user = self._request('GET', '/user', token)
        repo = self._request('GET', '/repos/' + repository, token)
        if (not isinstance(user, dict) or not isinstance(repo, dict)
                or not isinstance(user.get('login'), str)
                or not isinstance(repo.get('full_name'), str)
                or repo['full_name'].lower() != repository.lower()
                or not isinstance(repo.get('permissions'), dict)
                or repo['permissions'].get('push') is not True):
            self._fail('repository_push_access_required', 'github_read')
        return {'authenticated':True, 'repository':repository, 'login':user['login'],
                'repository_push_role':True, 'token_contents_write_verified':False,
                'scope_note':'Read-only check; actual object creation confirms Contents-write.'}

    def raw_blob(self, sha, size, token):
        token = self._token(token)
        if (not isinstance(sha,str) or re.fullmatch('[0-9a-f]{40}',sha) is None
                or type(size) is not int or not 0 <= size <= MAX_RAW_BLOB):
            self._fail('invalid_raw_blob_identity', 'git_blob_read')
        path = '/repos/' + self._repo(self.REPOSITORY) + '/git/blobs/' + sha
        with self._coordinator.lock:
            raw = self._perform('GET', path, token, accept='application/vnd.github.raw+json',
                                limit=size, binary=True)
            if len(raw) != size or self._git_sha(raw) != sha:
                self._fail('raw_blob_integrity_mismatch', 'git_blob_read')
            return raw


def wrap(legacy_api, *, _opener=None, _coordinator=None):
    """Return the compatible proxy; test injection never affects the shared default."""
    if getattr(legacy_api, '_e3_serial_transport', False):
        return legacy_api
    return _Proxy(legacy_api, _opener or build_opener(_NoRedirect()), _coordinator or _GLOBAL)
