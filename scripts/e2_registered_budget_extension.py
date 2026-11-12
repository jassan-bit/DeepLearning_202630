"""Prospective, explicitly authorized funding for the unchanged E2 ledger.

This does not change the original migration, old records, worker caps or model
code. A clean original ledger is sealed under its original64-hour allowance;
subsequent rows are linked to a separately registered user authorization.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading

import e2_reduced_budget as reduced

historical = reduced.historical
canonical, digest = historical.canonical, historical.digest
BudgetError = historical.BudgetError
BudgetReconciliationRequired = historical.BudgetReconciliationRequired
VERSION = 'e2-registered-budget-extension-v1'
ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
REGISTRATION_PATH = 'experiments/e2_budget_extension_20261112/registration.json'
ANCHOR_NAME = 'funding_extension.json'
ORIGINAL_INNER = historical.PilotBudgetLedger
_CONSTRUCTOR_LOCK = threading.RLock()
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def policy_draft(total_hours=72):
    """Return a proposed document which cannot authorize execution."""
    if type(total_hours) is not int or total_hours not in (72,80):
        raise ValueError('Choose an explicit new TOTAL of72 or80hours')
    value={'version':VERSION,'engine_commit':ENGINE_COMMIT,'initial_total_hours':64,
        'authorized_total_hours':total_hours,'added_seconds':(total_hours-64)*3600,
        'budget_reset':False,'worker_caps_changed':False,'status':'proposed_not_authorized',
        'authorization':None}
    value['sha256']=digest(value)
    return value


def validate_funding(policy):
    historical._checked(policy)
    value=deepcopy(policy)
    total=value.get('authorized_total_hours')
    if (value.get('version')!=VERSION or value.get('engine_commit')!=ENGINE_COMMIT
            or value.get('initial_total_hours')!=64 or type(total)is not int or total not in (72,80)
            or value.get('added_seconds')!=(total-64)*3600 or value.get('budget_reset')is not False
            or value.get('worker_caps_changed')is not False or value.get('status')!='authorized_by_user'):
        raise BudgetError('explicit_registered_funding_authorization_required')
    authority=value.get('authorization')
    if (not isinstance(authority,dict) or authority.get('source')!='explicit_user_confirmation'
            or authority.get('approved_total_hours')!=total
            or not HEX64.fullmatch(str(authority.get('evidence_sha256','')))):
        raise BudgetError('explicit_user_total_and_evidence_required')
    return value


def verify_funding_registration(policy, commit, git_root, relative_path=REGISTRATION_PATH):
    value=validate_funding(policy)
    if not isinstance(commit,str) or not re.fullmatch(r'[0-9a-f]{40}',commit):
        raise BudgetError('actual_funding_registration_commit_required')
    part=Path(relative_path)
    if part.is_absolute() or '..'in part.parts:
        raise BudgetError('funding_registration_path_must_be_relative')
    try:
        raw=subprocess.check_output(['git','-C',str(git_root),'show',commit+':'+relative_path],stderr=subprocess.DEVNULL)
        stamp=subprocess.check_output(['git','-C',str(git_root),'show','-s','--format=%cI',commit],stderr=subprocess.DEVNULL).decode().strip()
    except (OSError,subprocess.CalledProcessError):
        raise BudgetError('funding_registration_object_missing_or_invalid') from None
    record=json.loads(raw)
    actual=datetime.fromisoformat(stamp.replace('Z','+00:00'))
    if (record.get('funding_policy')!=value or record.get('engine_commit')!=ENGINE_COMMIT
            or record.get('protocol_id')!='e2-reduced-delivery-v1'
            or actual.tzinfo is None or actual>datetime.now(timezone.utc)):
        raise BudgetError('actual_prior_git_funding_binding_required')
    return {'registration_commit':commit,'registration_path':relative_path,
        'registration_file_sha256':hashlib.sha256(raw).hexdigest(),
        'registered_at_utc':actual.astimezone(timezone.utc).isoformat(),
        'funding_policy_sha256':value['sha256']}


def _read_ledger(path):
    raw=Path(path).read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise BudgetReconciliationRequired('existing_complete_base_ledger_required')
    return raw


def _replay64(raw,reconciliation):
    """Read only; original class/allowance validates the entire old prefix."""
    reader=object.__new__(ORIGINAL_INNER)
    reader._events,reader._active,reader._charges=[],{},0.
    reader._attempts,reader._progress=set(),{}
    reader.reconciliation=deepcopy(reconciliation)
    for index,line in enumerate(raw.splitlines()):
        row=json.loads(line,object_pairs_hook=historical._pairs)
        checked=historical._checked(row)
        if (checked.get('version')!=historical.VERSION or checked.get('sequence')!=index
                or checked.get('reconciliation_sha256')!=reconciliation['sha256']
                or checked.get('previous_sha256')!=(reader._events[-1]['sha256']if index else historical.ZERO)
                or 'funding_policy_sha256'in checked):
            raise BudgetReconciliationRequired('base64_prefix_chain_or_funding_changed')
        if checked.get('kind')=='charged' and checked.get('status')not in('completed','failed','paused_clean'):
            raise BudgetReconciliationRequired('unknown_base_attempt_preserves_reservation')
        reader._apply(row)
    if reader._active:
        raise BudgetReconciliationRequired('active_base_lease_cannot_receive_funding')
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
        'events':len(reader._events),'last_event_sha256':reader._events[-1]['sha256'],
        'charged_seconds':reader._charges}


def make_funded_budget_class(funding_policy, *, funding_registration_commit,
        funding_git_root, funding_registration_path=REGISTRATION_PATH):
    policy=validate_funding(funding_policy)
    proof=verify_funding_registration(policy,funding_registration_commit,funding_git_root,funding_registration_path)

    class FundedHistoricalLedger(ORIGINAL_INNER):
        def __init__(self,root,reconciliation):
            self._funding_active=False
            self._funding_anchor=None
            self._funding_policy=policy
            self._funding_proof=proof
            self._anchor_path=Path(root)/ANCHOR_NAME
            if self._anchor_path.exists():
                self._funding_anchor=historical._read(self._anchor_path)
                self._verify_anchor(reconciliation)
            super().__init__(root,reconciliation)
            try:
                # The original inner constructor holds the canonical writer
                # lock and rejects active leases before publishing the anchor.
                if self._funding_anchor is None:
                    if not (self.root/'reduced_migration.json').is_file():
                        raise BudgetReconciliationRequired('existing_reduced_migration_required')
                    prefix=_replay64(_read_ledger(self._path),reconciliation)
                    anchor={'version':VERSION,'funding_policy_sha256':policy['sha256'],
                        'registration_proof':proof,'historical_reconciliation_sha256':reconciliation['sha256'],
                        'initial_total_hours':64,'effective_total_hours':policy['authorized_total_hours'],
                        'base_prefix':prefix,'budget_reset':False}
                    anchor['sha256']=digest(anchor)
                    reduced._write_new(self._anchor_path,anchor)
                    self._funding_anchor=anchor
                self._verify_anchor(reconciliation)
                self._funding_active=True
            except BaseException:
                self.close();raise

        def _verify_anchor(self,reconciliation):
            a=historical._checked(self._funding_anchor)
            if (a.get('version')!=VERSION or a.get('funding_policy_sha256')!=policy['sha256']
                    or a.get('registration_proof')!=proof or a.get('budget_reset')is not False
                    or a.get('initial_total_hours')!=64 or a.get('effective_total_hours')!=policy['authorized_total_hours']
                    or a.get('historical_reconciliation_sha256')!=reconciliation['sha256']):
                raise BudgetReconciliationRequired('funding_anchor_changed_no_extra_credit')
            raw=_read_ledger(Path(self._anchor_path).parent/'pilot_budget.jsonl')
            prefix=a['base_prefix'];old=raw[:prefix['bytes']]
            if len(old)!=prefix['bytes'] or hashlib.sha256(old).hexdigest()!=prefix['sha256']:
                raise BudgetReconciliationRequired('base64_prefix_bytes_changed')
            if _replay64(old,reconciliation)!=prefix:
                raise BudgetReconciliationRequired('base64_prefix_receipt_changed')

        @property
        def remaining_seconds(self):
            extra=policy['added_seconds']if getattr(self,'_funding_active',False)else 0
            return max(0.,self.reconciliation['remaining_seconds_before_pilots']+extra
                -self._charges-sum(self._active.values()))

        def _apply(self,record):
            # Inherited-ledger readers constructed without __init__ retain64.
            anchor=getattr(self,'_funding_anchor',None)
            if anchor is not None:
                funded=record.get('sequence',-1)>=anchor['base_prefix']['events']
                if funded and(record.get('funding_policy_sha256')!=policy['sha256']
                        or record.get('funding_registration_commit')!=funding_registration_commit):
                    raise BudgetReconciliationRequired('funded_row_missing_immutable_authority')
                if not funded and'funding_policy_sha256'in record:
                    raise BudgetReconciliationRequired('retroactive_funding_not_allowed')
                self._funding_active=funded
            super()._apply(record)

        def _append(self,kind,**fields):
            if self._closed:raise BudgetError('pilot_budget_ledger_closed')
            fields.update(funding_policy_sha256=policy['sha256'],
                funding_registration_commit=funding_registration_commit,
                effective_total_hours=policy['authorized_total_hours'])
            row={'version':historical.VERSION,'sequence':len(self._events),'kind':kind,
                'reconciliation_sha256':self.reconciliation['sha256'],
                'previous_sha256':self._events[-1]['sha256']if self._events else historical.ZERO,
                'utc':datetime.now(timezone.utc).isoformat(),**fields}
            row['sha256']=digest(row)
            clone=object.__new__(type(self))
            clone._events,clone._active=list(self._events),dict(self._active)
            clone._charges,clone._attempts=self._charges,set(self._attempts)
            clone._progress=deepcopy(self._progress);clone.reconciliation=self.reconciliation
            clone._funding_anchor=self._funding_anchor;clone._funding_active=self._funding_active
            clone._apply(row)
            with historical._no_links(self._path).open('ab')as stream:
                stream.write(canonical(row)+b'\n');stream.flush()
                import os
                os.fsync(stream.fileno())
            self._apply(row)
            return deepcopy(row)

    class RegisteredFundedBudget(reduced.ReducedBudgetLedger):
        def __init__(self,*args,**kwargs):
            # Patch only construction, and use our own _append implementation
            # thereafter. The original module and its source bytes stay intact.
            with _CONSTRUCTOR_LOCK:
                previous=historical.PilotBudgetLedger
                historical.PilotBudgetLedger=FundedHistoricalLedger
                try:super().__init__(*args,**kwargs)
                finally:historical.PilotBudgetLedger=previous
        def report(self):
            result=super().report()
            return {**result,'initial_authorized_total_hours':64,
                'effective_authorized_total_hours':policy['authorized_total_hours'],
                'authorized_extension_seconds':policy['added_seconds'],
                'funding_policy_sha256':policy['sha256'],
                'funding_registration_commit':funding_registration_commit,
                'funding_base_prefix':deepcopy(self._inner._funding_anchor['base_prefix']),
                'budget_reset':False}
    return RegisteredFundedBudget
