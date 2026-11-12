"""Prospective per-phase resource bounds without another grant of campaign time.

The original600-second migration and prior receipts remain unchanged. This
adapter is separately registered before new reservations and applies1800seconds
per future optimizer phase. The existing H funding proof still supplies72hours.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

VERSION = 'e2-registered-resource-extension-v1'
ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
FUNDING_COMMIT = 'f540227c23c572b9ca7fb36920588727b029484a'
REGISTRATION_PATH = 'experiments/e2_resource_continuation_20261112/registration.json'
SOURCE_PATH = 'scripts/e2_registered_resource_extension.py'
ANCHOR_NAME = 'resource_extension.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked(value):
    if not isinstance(value, dict) or value.get('sha256') != digest({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('resource_policy_or_anchor_checksum_changed')
    return deepcopy(value)


def validate_resource_policy(policy):
    value = _checked(policy)
    required = {'version':VERSION, 'engine_commit':ENGINE_COMMIT,
        'funding_registration_commit':FUNDING_COMMIT,
        'initial_phase_wall_seconds':600, 'stage_wall_seconds':1800,
        'worker_overhead_seconds':60, 'max_optimization_phases':3,
        'max_epochs':200, 'patience_epochs':20, 'authorized_total_hours':72,
        'budget_reset':False, 'new_opportunities_only':True,
        'scientific_retry_allowed':False, 'models_seeds_data_unchanged':True,
        'prior_failed_attempts_retried':False}
    for key, expected in required.items():
        actual=value.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('registered_prospective_resource_policy_required:'+key)
    return value


def verify_resource_registration(policy, registration_commit, registration_root, extension_root=None):
    policy = validate_resource_policy(policy)
    if not isinstance(registration_commit, str) or not re.fullmatch(r'[0-9a-f]{40}',registration_commit):
        raise ValueError('actual_resource_registration_commit_required')
    def git(*args):
        return subprocess.check_output(['git','-C',str(registration_root),*args],stderr=subprocess.DEVNULL)
    try:
        raw=git('show',registration_commit+':'+REGISTRATION_PATH)
        source=git('show',registration_commit+':'+SOURCE_PATH)
        stamp=git('show','-s','--format=%cI',registration_commit).decode().strip()
    except (OSError,subprocess.CalledProcessError):
        raise ValueError('resource_registration_object_missing') from None
    record=json.loads(raw);actual=datetime.fromisoformat(stamp.replace('Z','+00:00'))
    expected=record.get('files_sha256',{}).get(SOURCE_PATH)
    if (record.get('protocol_id')!='e2-reduced-delivery-v1'
            or record.get('engine_commit')!=ENGINE_COMMIT or record.get('resource_policy')!=policy
            or actual.tzinfo is None or actual>datetime.now(timezone.utc)
            or expected!=hashlib.sha256(source).hexdigest() or file_sha(__file__)!=expected):
        raise ValueError('actual_prior_resource_git_binding_required')
    if extension_root is not None:
        runtime=Path(extension_root)/SOURCE_PATH
        if runtime.is_symlink() or not runtime.resolve().is_relative_to(Path(extension_root).resolve()) or file_sha(runtime)!=expected:
            raise ValueError('resource_runtime_source_changed')
    return {'registration_commit':registration_commit,'registration_path':REGISTRATION_PATH,
        'registration_file_sha256':hashlib.sha256(raw).hexdigest(),
        'registered_at_utc':actual.astimezone(timezone.utc).isoformat(),
        'source_sha256':expected,'resource_policy_sha256':policy['sha256']}


def _ledger_prefix(path):
    raw=Path(path).read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise ValueError('complete_existing_resource_ledger_required')
    rows=[];previous='0'*64
    for index,line in enumerate(raw.splitlines()):
        row=json.loads(line)
        if (row.get('sequence')!=index or row.get('previous_sha256')!=previous
                or row.get('sha256')!=digest({k:v for k,v in row.items() if k!='sha256'})):
            raise ValueError('existing_resource_ledger_chain_changed')
        rows.append(row);previous=row['sha256']
    return raw,rows


def make_resource_budget_class(funded_constructor, resource_policy, *, registration_commit,
        registration_root, extension_root=None):
    """Return a funded-ledger subclass; no live process or source is patched."""
    policy=validate_resource_policy(resource_policy)
    proof=verify_resource_registration(policy,registration_commit,registration_root,extension_root)
    if not isinstance(funded_constructor,type):
        raise TypeError('a_registered_funded_budget_class_is_required')

    class RegisteredResourceBudget(funded_constructor):
        def __init__(self,*args,**kwargs):
            self._resource_anchor=None
            super().__init__(*args,**kwargs)
            try:
                funding=super().report()
                if (funding.get('effective_authorized_total_hours')!=72
                        or funding.get('funding_registration_commit')!=FUNDING_COMMIT
                        or self.migration.get('stage_wall_seconds')!=600):
                    raise ValueError('original_H_funding_and_migration_must_remain_unchanged')
                funding_path=self.root/'funding_extension.json'
                self._resource_funding_sha=file_sha(funding_path)
                self._resource_anchor_path=self.root/ANCHOR_NAME
                raw,rows=_ledger_prefix(self.root/'pilot_budget.jsonl')
                if self._resource_anchor_path.exists():
                    self._resource_anchor=_checked(json.loads(self._resource_anchor_path.read_bytes()))
                else:
                    if any(r.get('metadata',{}).get('resource_policy_sha256') for r in rows if r.get('kind')=='reserved'):
                        raise ValueError('existing_resource_rows_cannot_be_adopted_without_their_anchor')
                    anchor={'version':VERSION,'resource_policy_sha256':policy['sha256'],
                        'registration_proof':proof,'funding_registration_commit':FUNDING_COMMIT,
                        'funding_anchor_file_sha256':self._resource_funding_sha,
                        'initial_phase_wall_seconds':600,'prospective_phase_wall_seconds':1800,
                        'effective_total_hours':72,'budget_reset':False,
                        'prior_ledger_prefix':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                            'events':len(rows),'last_event_sha256':rows[-1]['sha256']}}
                    anchor['sha256']=digest(anchor)
                    with self._resource_anchor_path.open('xb') as stream:
                        stream.write(canonical(anchor)+b'\n');stream.flush()
                        import os
                        os.fsync(stream.fileno())
                    self._resource_anchor=anchor
                self._verify_resource_anchor(raw,rows)
            except BaseException:
                self.close();raise

        def _verify_resource_anchor(self,raw=None,rows=None):
            anchor=_checked(self._resource_anchor)
            expected={'version':VERSION,'resource_policy_sha256':policy['sha256'],
                'registration_proof':proof,'funding_registration_commit':FUNDING_COMMIT,
                'funding_anchor_file_sha256':self._resource_funding_sha,
                'initial_phase_wall_seconds':600,'prospective_phase_wall_seconds':1800,
                'effective_total_hours':72,'budget_reset':False}
            if any(anchor.get(k)!=v for k,v in expected.items()):
                raise ValueError('resource_anchor_changed_no_reinterpretation')
            if file_sha(self.root/'funding_extension.json')!=self._resource_funding_sha:
                raise ValueError('original_funding_anchor_changed')
            if raw is None:raw,rows=_ledger_prefix(self.root/'pilot_budget.jsonl')
            prefix=anchor['prior_ledger_prefix'];old=raw[:prefix['bytes']]
            if (len(old)!=prefix['bytes'] or hashlib.sha256(old).hexdigest()!=prefix['sha256']
                    or len(old.splitlines())!=prefix['events']
                    or json.loads(old.splitlines()[-1])['sha256']!=prefix['last_event_sha256']):
                raise ValueError('old_resource_ledger_prefix_changed')
            for row in rows[prefix['events']:]:
                if row.get('kind')=='reserved':
                    m=row.get('metadata',{})
                    phases=m.get('resource_optimization_phases')
                    if (m.get('resource_policy_sha256')!=policy['sha256']
                            or m.get('resource_registration_commit')!=registration_commit
                            or type(phases)is not int or phases not in(1,2,3)
                            or row.get('reservation_seconds')!=self.reservation_for(phases)):
                        raise ValueError('new_resource_reservation_missing_its_registered_authority')

        @property
        def stage_wall_seconds(self):
            return policy['stage_wall_seconds']

        def reserve(self,attempt_id,seconds,*,metadata=None):
            self._verify_resource_anchor()
            amounts={self.reservation_for(n):n for n in(1,2,3)}
            if type(seconds)not in(int,float) or seconds not in amounts:
                raise ValueError('prospective_whole_worker_reservation_required')
            phases=amounts[seconds];data={}if metadata is None else deepcopy(metadata)
            if 'phases'in data and data['phases']!=phases:
                raise ValueError('resource_phase_count_and_reservation_differ')
            data.update(resource_policy_sha256=policy['sha256'],
                resource_registration_commit=registration_commit,
                initial_phase_wall_seconds=600, prospective_phase_wall_seconds=1800,
                resource_optimization_phases=phases)
            return super().reserve(attempt_id,seconds,metadata=data)

        def report(self):
            report=super().report()
            return {**report,'initial_phase_wall_seconds':600,
                'prospective_phase_wall_seconds':1800,
                'resource_policy_sha256':policy['sha256'],
                'resource_registration_commit':registration_commit,
                'resource_total_budget_extended':False,
                'effective_authorized_total_hours':72}
    return RegisteredResourceBudget
