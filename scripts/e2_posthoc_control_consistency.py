"""Read fresh GitHub refs and confirm only an identical posthoc publication.

This transport-only adapter never adopts another controller's state, changes a
budget, starts a worker, retries a fit, or forces a branch update. Scientific
sources stay unchanged. A single repeated publication is permitted only after
the exact expected base is observed again; ambiguous success is accepted only
when the complete remote state is identical to the proposed state.
"""
from __future__ import annotations

from functools import wraps
import json
from urllib.request import Request
import uuid

VERSION = 'e2-posthoc-control-consistency-v1'
MARKER = '_e2_posthoc_control_consistency_v1'
CAS_CODE = 'another_session_advanced_control_state'


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


class _FreshReads:
    """Add cache directives to API GETs without changing auth or redirects."""
    def __init__(self, delegate):
        self.delegate = delegate

    def open(self, request, *args, **kwargs):
        if isinstance(request, Request) and request.get_method() == 'GET':
            request.add_header('Cache-Control', 'no-cache, no-store, max-age=0')
            request.add_header('Pragma', 'no-cache')
        return self.delegate.open(request, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __repr__(self):
        return '_FreshReads(delegate=<existing authenticated opener>)'


def _fresh_opener(client):
    if not isinstance(client._opener, _FreshReads):
        client._opener = _FreshReads(client._opener)


def bind_state(state, backup_module):
    """Wrap one already-created state; never restore/adopt a foreign state."""
    if getattr(state, MARKER, False):
        return
    original_publish = state._publish
    pause = backup_module.BackupPause

    @wraps(original_publish)
    def publish(proposed, delta):
        # Freeze exact bytes, not just a caller-supplied state identifier.
        expected = None if state._state is None else _canonical(state._state)
        proposed_bytes = _canonical(proposed)
        try:
            return original_publish(proposed, delta)
        except pause as error:
            if getattr(error, 'code', None) != CAS_CODE:
                raise
            mismatch = error
        # At most two extra reads and one retry of the SAME publication.
        # A genuinely different state is never adopted, including a state
        # sharing an identifier but carrying different bytes.
        for attempt in range(2):
            head = state.client._branch_head()
            current = state._current(head)
            current_bytes = None if current is None else _canonical(current)
            if current_bytes == proposed_bytes:
                state._state = proposed
                return head
            if current_bytes != expected:
                raise mismatch
            if attempt:
                raise mismatch
            try:
                return original_publish(proposed, delta)
            except pause as error:
                if getattr(error, 'code', None) != CAS_CODE:
                    raise
                mismatch = error
        raise mismatch

    state._publish = publish
    setattr(state, MARKER, True)


def install(backup_module, old_module):
    """Patch only network/ref publication plumbing, idempotently per class."""
    backup_class = backup_module.GitHubBackup
    if not backup_class.__dict__.get(MARKER, False):
        original_init = backup_class.__init__

        @wraps(original_init)
        def initialize(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _fresh_opener(self)

        def branch_head(self):
            _fresh_opener(self)
            path = '/git/ref/heads/' + backup_module.BRANCH
            result = self._request('GET', path + '?e2_refresh=' + uuid.uuid4().hex)
            return result['object']['sha']

        backup_class.__init__ = initialize
        backup_class._branch_head = branch_head
        setattr(backup_class, MARKER, True)

    remote_class = old_module.PosthocRemoteState
    if not remote_class.__dict__.get(MARKER, False):
        original_remote_init = remote_class.__init__

        @wraps(original_remote_init)
        def initialize_remote(self, *args, **kwargs):
            original_remote_init(self, *args, **kwargs)
            _fresh_opener(self.state.client)
            bind_state(self.state, backup_module)

        remote_class.__init__ = initialize_remote
        setattr(remote_class, MARKER, True)

    return {'version': VERSION, 'fit_retry_performed': False,
            'unknown_control_state_adopted': False, 'budget_modified': False}
