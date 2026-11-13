"""An explicitly confirmed 80-hour continuation of the immutable H/P ledger.

The existing funding and resource anchors remain byte-for-byte unchanged. A
separate anchor seals the entire 72-hour journal before the extra eight hours
can be used. Old records replay under their original allowance; subsequent
records retain H's authority and add this continuation's consent and proof.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

# Approval construction also runs in Colab's host kernel, without importing
# the isolated scientific environment. Budget imports are deferred to factory.
BudgetError = ValueError
BudgetReconciliationRequired = ValueError

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()
VERSION = 'e2-confirmed-budget-continuation-v1'
ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
FUNDING_COMMIT = 'f540227c23c572b9ca7fb36920588727b029484a'
FUNDING_SHA = 'a1cf19c7a4c88f525afa5d7369546b4e31a898447cee56830ec520bd542e350f'
RECONCILIATION_SHA = 'fdcb6f63a059eb15311f399fbc5034c02bbf95d5b43fc9d19917f97988720b13'
PROPOSAL_PATH = 'experiments/e2_budget_continuation_20261113/proposal.json'
REGISTRATION_PATH = 'experiments/e2_budget_continuation_20261113/registration.json'
SOURCE_PATH = 'scripts/e2_confirmed_budget_continuation.py'
PROPOSAL_SHA = 'decf2c4d1adfa4f1ac7b4b29cd351fb24295031eab49e695c2a50f5a6e96101f'
ANCHOR_NAME = 'budget_continuation_80.json'
APPROVAL_NAME = 'budget_continuation_80_approval.json'


def _checked(value):
    if not isinstance(value, dict) or value.get('sha256') != digest({k:v for k,v in value.items() if k != 'sha256'}):
        raise BudgetError('continuation_checksum_changed')
    return deepcopy(value)


def _utc(value):
    if not isinstance(value, str):
        raise BudgetError('dated_explicit_confirmation_required')
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise BudgetError('dated_explicit_confirmation_required') from None
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise BudgetError('dated_explicit_confirmation_required')
    return stamp


def validate_proposal(proposal):
    value = _checked(proposal)
    required = {'version': VERSION, 'engine_commit': ENGINE_COMMIT,
        'protocol_id': 'e2-reduced-delivery-v1', 'previous_total_hours': 72,
        'proposed_total_hours': 80, 'added_seconds': 28800,
        'prior_funding_registration_commit': FUNDING_COMMIT,
        'prior_funding_policy_sha256': FUNDING_SHA,
        'historical_reconciliation_sha256': RECONCILIATION_SHA,
        'status': 'proposed_not_authorized',
        'authorization_source_required': 'explicit_colab_confirmation',
        'budget_reset': False, 'worker_caps_changed': False,
        'models_seeds_data_unchanged': True, 'scientific_retry_allowed': False,
        'persistent_destination': 'github', 'sha256': PROPOSAL_SHA}
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in required.items()):
        raise BudgetError('the_registered_72_to_80_proposal_is_required')
    return value


def build_approval_receipt(*, continuation_commit, confirmed_text, observed_at_utc=None):
    """Call only with the value actually returned by the Colab input prompt."""
    if confirmed_text != '80' or not isinstance(continuation_commit, str) or not re.fullmatch(r'[0-9a-f]{40}', continuation_commit):
        raise BudgetError('explicit_colab_input_80_required')
    stamp = observed_at_utc or datetime.now(timezone.utc).isoformat()
    _utc(stamp)
    value = {'version': VERSION, 'source': 'explicit_colab_confirmation',
        'confirmed_text': confirmed_text, 'approved_total_hours': 80,
        'previous_total_hours': 72, 'added_seconds': 28800,
        'budget_reset': False, 'proposal_sha256': PROPOSAL_SHA,
        'continuation_registration_commit': continuation_commit,
        'observed_at_utc': stamp}
    value['sha256'] = digest(value)
    return value


def validate_approval(receipt, registration_commit):
    value = _checked(receipt)
    required = {'version': VERSION, 'source': 'explicit_colab_confirmation',
        'confirmed_text': '80', 'approved_total_hours': 80,
        'previous_total_hours': 72, 'added_seconds': 28800,
        'budget_reset': False, 'proposal_sha256': PROPOSAL_SHA,
        'continuation_registration_commit': registration_commit}
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in required.items()):
        raise BudgetError('explicit_colab_80_confirmation_bound_to_this_proposal_required')
    _utc(value.get('observed_at_utc'))
    return value


def _registration(proposal, commit, root):
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise BudgetError('actual_continuation_registration_commit_required')
    def git(path):
        return subprocess.check_output(['git', '-C', str(root), 'show', commit + ':' + path], stderr=subprocess.DEVNULL)
    try:
        raw, published = git(REGISTRATION_PATH), git(PROPOSAL_PATH)
        source = git(SOURCE_PATH)
        stamp = subprocess.check_output(['git', '-C', str(root), 'show', '-s', '--format=%cI', commit], stderr=subprocess.DEVNULL).decode().strip()
        record = json.loads(raw)
        _utc(stamp)
    except (OSError, ValueError, subprocess.CalledProcessError):
        raise BudgetError('actual_continuation_registration_object_required') from None
    expected = record.get('files_sha256', {}).get(SOURCE_PATH)
    if (json.loads(published) != proposal or record.get('proposal_sha256') != hashlib.sha256(published).hexdigest()
            or record.get('proposal_policy_sha256') != PROPOSAL_SHA
            or record.get('protocol_id') != 'e2-reduced-delivery-v1'
            or record.get('engine_commit') != ENGINE_COMMIT
            or expected != hashlib.sha256(source).hexdigest()
            or expected != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise BudgetError('continuation_registration_source_or_proposal_changed')
    return {'registration_commit': commit, 'registration_path': REGISTRATION_PATH,
        'registration_file_sha256': hashlib.sha256(raw).hexdigest(),
        'proposal_file_sha256': hashlib.sha256(published).hexdigest(),
        'source_sha256': expected}


def _prefix(path):
    raw = historical._no_links(path).read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise BudgetReconciliationRequired('complete_existing_72_hour_ledger_required')
    rows = [json.loads(line, object_pairs_hook=historical._pairs) for line in raw.splitlines()]
    previous = historical.ZERO
    for index, row in enumerate(rows):
        _checked(row)
        if row.get('sequence') != index or row.get('previous_sha256') != previous:
            raise BudgetReconciliationRequired('continuation_existing_journal_chain_changed')
        previous = row['sha256']
    return raw, rows


def make_continued_budget_class(base_class, proposal, *, registration_commit,
        registration_git_root, approval_receipt):
    """Extend a fresh, pinned H/P constructor without changing its source.

    H keeps its inner class in the constructor closure. The pinned layout is
    checked explicitly; a different implementation fails closed. Only that
    uniquely created H class's construction is routed, under its existing lock,
    to an inner subclass. All H/P initialization and anchor checks still run.
    """
    global funding, historical, reduced, BudgetError, BudgetReconciliationRequired
    import e2_registered_budget_extension as funding
    historical, reduced = funding.historical, funding.reduced
    BudgetError, BudgetReconciliationRequired = historical.BudgetError, historical.BudgetReconciliationRequired
    proposal = validate_proposal(proposal)
    approval = validate_approval(approval_receipt, registration_commit)
    proof = _registration(proposal, registration_commit, registration_git_root)
    candidates = [c for c in base_class.__mro__ if c.__name__ == 'RegisteredFundedBudget'
        and c.__module__ == funding.__name__]
    if len(candidates) != 1:
        raise BudgetError('fresh_pinned_H_P_budget_constructor_required')
    funded_class = candidates[0]
    original_constructor = funded_class.__dict__.get('__init__')
    cells = dict(zip(original_constructor.__code__.co_freevars,
        [cell.cell_contents for cell in original_constructor.__closure__ or ()]))
    original_inner = cells.get('FundedHistoricalLedger')
    if not isinstance(original_inner, type) or not issubclass(original_inner, funding.ORIGINAL_INNER):
        raise BudgetError('pinned_H_inner_constructor_layout_changed')

    class ContinuedInner(original_inner):
        def __init__(self, root, reconciliation):
            self._continuation_anchor = None
            self._continuation_active = False
            self._continuation_root = historical._no_links(Path(root))
            self._continuation_path = self._continuation_root / ANCHOR_NAME
            if self._continuation_path.exists():
                self._continuation_anchor = historical._read(self._continuation_path)
                self._verify_continuation(reconciliation)
            super().__init__(root, reconciliation)
            try:
                if (self._funding_policy.get('sha256') != FUNDING_SHA
                        or self._funding_proof.get('registration_commit') != FUNDING_COMMIT
                        or reconciliation.get('sha256') != RECONCILIATION_SHA
                        or self._active):
                    raise BudgetReconciliationRequired('clean_unchanged_H_72_hour_budget_required')
                if self._continuation_anchor is None:
                    raw, rows = _prefix(self._path)
                    if any('continuation_policy_sha256' in r for r in rows):
                        raise BudgetReconciliationRequired('continuation_rows_without_their_original_anchor')
                    value = {'version': VERSION, 'proposal_sha256': PROPOSAL_SHA,
                        'registration_proof': proof, 'approval_receipt': approval,
                        'continuation_registration_commit': registration_commit,
                        'approval_sha256': approval['sha256'],
                        'historical_reconciliation_sha256': RECONCILIATION_SHA,
                        'previous_total_hours': 72, 'effective_total_hours': 80,
                        'added_seconds': 28800, 'budget_reset': False,
                        'funding_anchor_file_sha256': hashlib.sha256((self.root / 'funding_extension.json').read_bytes()).hexdigest(),
                        'resource_anchor_file_sha256': hashlib.sha256((self.root / 'resource_extension.json').read_bytes()).hexdigest(),
                        'prior_ledger_prefix': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                            'events': len(rows), 'last_event_sha256': rows[-1]['sha256']}}
                    value['sha256'] = digest(value)
                    # Consent is persisted before the anchor and before any
                    # continued overhead, reservation or optimizer execution.
                    consent_path = self.root / APPROVAL_NAME
                    if consent_path.exists():
                        if historical._read(consent_path) != approval:
                            raise BudgetReconciliationRequired('existing_continuation_consent_differs')
                    else:
                        reduced._write_new(consent_path, approval)
                    reduced._write_new(self._continuation_path, value)
                    self._continuation_anchor = value
                self._verify_continuation(reconciliation)
                self._continuation_active = True
            except BaseException:
                self.close()
                raise

        def _verify_continuation(self, reconciliation):
            a = _checked(self._continuation_anchor)
            required = {'version': VERSION, 'proposal_sha256': PROPOSAL_SHA,
                'registration_proof': proof, 'historical_reconciliation_sha256': RECONCILIATION_SHA,
                'previous_total_hours': 72, 'effective_total_hours': 80,
                'added_seconds': 28800, 'budget_reset': False}
            if any(a.get(k) != v for k, v in required.items()) or reconciliation.get('sha256') != RECONCILIATION_SHA:
                raise BudgetReconciliationRequired('continuation_anchor_authority_changed')
            stored = validate_approval(a.get('approval_receipt'), registration_commit)
            if a.get('continuation_registration_commit') != registration_commit or a.get('approval_sha256') != stored['sha256']:
                raise BudgetReconciliationRequired('continuation_anchor_consent_binding_changed')
            if historical._read(self._continuation_root / APPROVAL_NAME) != stored:
                raise BudgetReconciliationRequired('stored_continuation_consent_changed')
            for name, key in [('funding_extension.json', 'funding_anchor_file_sha256'),
                    ('resource_extension.json', 'resource_anchor_file_sha256')]:
                if hashlib.sha256(historical._no_links(self._continuation_root / name).read_bytes()).hexdigest() != a.get(key):
                    raise BudgetReconciliationRequired('original_H_P_anchors_changed')
            raw, rows = _prefix(self._continuation_root / 'pilot_budget.jsonl')
            prefix = a['prior_ledger_prefix']
            old = raw[:prefix['bytes']]
            if (len(old) != prefix['bytes'] or hashlib.sha256(old).hexdigest() != prefix['sha256']
                    or len(old.splitlines()) != prefix['events']
                    or json.loads(old.splitlines()[-1])['sha256'] != prefix['last_event_sha256']):
                raise BudgetReconciliationRequired('original_72_hour_prefix_bytes_changed')
            for row in rows:
                self._check_continuation_row(row)

        def _check_continuation_row(self, row):
            a = self._continuation_anchor
            continued = a is not None and row.get('sequence', -1) >= a['prior_ledger_prefix']['events']
            if continued:
                required = {'continuation_policy_sha256': PROPOSAL_SHA,
                    'continuation_registration_commit': registration_commit,
                    'continuation_approval_sha256': a['approval_receipt']['sha256'],
                    'effective_total_hours': 80}
                if any(row.get(k) != v for k, v in required.items()):
                    raise BudgetReconciliationRequired('continued_row_missing_immutable_confirmation')
            elif any(k.startswith('continuation_') for k in row):
                raise BudgetReconciliationRequired('retroactive_continuation_not_allowed')
            return continued

        @property
        def remaining_seconds(self):
            # Do not add credit to H's already-clipped remaining_seconds: that
            # would incorrectly grant extra time after the old cap was spent.
            old = self._funding_policy['added_seconds'] if getattr(self, '_funding_active', False) else 0
            extra = 28800 if getattr(self, '_continuation_active', False) else 0
            return max(0., self.reconciliation['remaining_seconds_before_pilots'] + old + extra
                - self._charges - sum(self._active.values()))

        def _apply(self, record):
            self._continuation_active = self._check_continuation_row(record)
            return super()._apply(record)

        def _append(self, kind, **fields):
            if self._closed:
                raise BudgetError('pilot_budget_ledger_closed')
            fields.update(funding_policy_sha256=self._funding_policy['sha256'],
                funding_registration_commit=self._funding_proof['registration_commit'],
                continuation_policy_sha256=PROPOSAL_SHA,
                continuation_registration_commit=registration_commit,
                continuation_approval_sha256=self._continuation_anchor['approval_receipt']['sha256'],
                effective_total_hours=80)
            row = {'version': historical.VERSION, 'sequence': len(self._events), 'kind': kind,
                'reconciliation_sha256': self.reconciliation['sha256'],
                'previous_sha256': self._events[-1]['sha256'] if self._events else historical.ZERO,
                'utc': datetime.now(timezone.utc).isoformat(), **fields}
            row['sha256'] = digest(row)
            clone = object.__new__(type(self))
            clone.__dict__ = self.__dict__.copy()
            clone._events, clone._active = list(self._events), dict(self._active)
            clone._attempts, clone._progress = set(self._attempts), deepcopy(self._progress)
            clone._apply(row)
            with historical._no_links(self._path).open('ab') as stream:
                stream.write(canonical(row) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
            self._apply(row)
            return deepcopy(row)

    class ConfirmedContinuationBudget(base_class):
        def __init__(self, *args, **kwargs):
            def construct(instance, *a, **kw):
                previous = historical.PilotBudgetLedger
                historical.PilotBudgetLedger = ContinuedInner
                try:
                    reduced.ReducedBudgetLedger.__init__(instance, *a, **kw)
                finally:
                    historical.PilotBudgetLedger = previous
            with funding._CONSTRUCTOR_LOCK:
                if funded_class.__dict__.get('__init__') is not original_constructor:
                    raise BudgetError('another_funding_constructor_adapter_is_active')
                funded_class.__init__ = construct
                try:
                    super().__init__(*args, **kwargs)
                finally:
                    funded_class.__init__ = original_constructor

        def report(self):
            report = super().report()
            a = self._inner._continuation_anchor
            return {**report, 'effective_authorized_total_hours': 80,
                'prior_effective_authorized_total_hours': 72,
                'continuation_authorized_extension_seconds': 28800,
                'continuation_policy_sha256': PROPOSAL_SHA,
                'continuation_registration_commit': registration_commit,
                'continuation_approval_sha256': a['approval_receipt']['sha256'],
                'continuation_base_prefix': deepcopy(a['prior_ledger_prefix']),
                'budget_reset': False}
    return ConfirmedContinuationBudget
