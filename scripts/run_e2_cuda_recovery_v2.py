"""Authorization-bound second-generation recovery of the sealed A3 campaign.

Inspection uses only the standard library. The scientific checkout, admission,
plan, data policy and isolated worker remain unchanged. A receipt on a separate
branch records the additional physical attempt; its predecessor stays unknown.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'e2-external-cuda-recovery-v2'
REPOSITORY = 'jassan-bit/DeepLearning_202630'
BRANCH = 'campaigns/e2-cuda-selection-v1'
SOURCE = '564ba4762f84a800b02fa4c734f5752a169f5223'
REGISTRATION = '78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9'
PLAN = '4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda'
UNKNOWN = {'outcome': 'unknown', 'elapsed_seconds': None, 'optimizer_steps': None,
           'optimizer_steps_known_lower_bound': 0,
           'partial_optimizer_update_may_have_occurred': True}
CONTROLLER_FILES = ('scripts/run_e2_cuda_recovery_v2.py',
                    'scripts/e2_manual_reconciliation_v2.py',
                    'scripts/e2_recovery_receipts_v2.py')
DECISION = 'manual_retry_same_sealed_logical_job_once_after_confirmed_executor_and_outcome_absent'


# Positive public exception codes from this controller and the pinned A3/R2 helpers.
# Arbitrary exception text, including valid-looking lower-case tokens, is private.
SAFE_ERROR_CODES = frozenset("""
github_rate_limit_pause
actual_outcome_job_identity_changed
actual_outcome_predictions_invalid
actual_outcome_status_invalid
admitted_job_has_no_remaining_time
authorization_binding_changed
authorization_budget_binding_invalid
authorization_hash_mismatch
authorization_incident_binding_invalid
authorization_incident_shape_invalid
campaign_branch_and_prefix_must_match
campaign_branch_required
campaign_budget_below_session_budget
campaign_commit_ack_missing
campaign_prefix_must_match_branch
campaign_record_directory_invalid
campaign_record_file_invalid
campaign_record_required
campaign_record_unavailable
campaign_recovery_reservation_spent_before_worker
campaign_recovery_reservation_unaffordable
campaign_remote_binding_changed
campaign_snapshot_limit
campaign_symlink_refused
candidate_hash_changed
cannot_reduce_an_existing_budget_extension
closed_recovery_implementation_required
completed_receipt_campaign_link_changed
completed_receipt_has_no_observed_outcome
completed_recovery_campaign_result_missing
controller_checkout_does_not_match_pin
controller_import_provenance_changed
cuda_unavailable
derived_status_plan_mismatch
duplicate_json_key
duplicate_recovery_session_close
event_filename_mismatch
existing_campaign_bytes_changed
explicit_retry_lost_once_required
external_report_already_exists
external_report_must_be_outside_campaign
external_report_persistence_failed
frozen_source_has_changes
git_archive_too_large
git_blob_invalid
git_blob_too_large
git_commit_message_refused
git_commit_parents_refused
git_object_invalid
git_object_type_mismatch
git_operation_failed
git_push_unconfirmed
git_remote_read_failed
git_remote_ref_mismatch
git_request_unconfirmed
git_transport_closed
git_transport_operation_refused
git_transport_url_refused
git_tree_entry_refused
git_tree_invalid
git_tree_path_refused
github_archive_content_invalid
github_archive_duplicate
github_archive_entry_invalid
github_archive_file_too_large
github_archive_invalid
github_archive_inventory_mismatch
github_archive_path_invalid
github_archive_root_invalid
github_archive_snapshot_mismatch
github_archive_too_large
github_archive_unavailable
github_blob_hash_mismatch
github_campaign_head_conflict
github_campaign_tree_missing
github_object_invalid
github_ref_ack_missing
github_refresh_required_after_failure
github_request_unconfirmed
github_response_invalid
github_response_too_large
github_secret_required
github_snapshot_capacity_pause
github_snapshot_invalid
github_snapshot_plan_mismatch
github_token_required
github_tree_invalid
github_uploaded_blob_mismatch
github_uploaded_snapshot_mismatch
immutable_controller_commit_required
immutable_remote_record_changed
immutable_source_commit_required
implementation_hash_mismatch
initial_budget_must_match_published_admission_use_explicit_extension
invalid_commit_message
invalid_cuda_search_admission
invalid_domain_hash
invalid_github_repository
invalid_implementation_manifest_path
invalid_phase_epochs
invalid_plan_identifier
invalid_plan_remote_storage
invalid_public_identity_hash
invalid_public_identity_string
invalid_public_job_identity
invalid_public_origin
invalid_public_phase_hash
invalid_public_resource_boolean
invalid_public_resource_count
invalid_public_resource_number
invalid_public_run_id
invalid_public_stage
invalid_public_trial_fields
invalid_reconciliation_accounting_boundary
invalid_retry_interval
invalid_scientific_json
invalid_snapshot_limit
invalid_snapshot_reservation
invalid_test_git_remote
invalid_trial_fatal_flag
invalid_trial_status
invalid_validation_predictions
local_git_remote_is_test_only
local_lock_check_unavailable
local_lock_path_invalid
local_metadata_invalid
local_remote_history_missing
local_remote_record_conflict
local_restore_target_appeared
locked_cuda_environment_required
locked_dependency_version_required
logical_finish_differs_from_physical_recovery
logical_result_differs_from_observed_retry
managed_python_3_12_12_required
manual_authorization_hash_invalid
manual_authorization_not_registered
manual_authorization_plan_changed
manual_budget_controls_invalid
manual_campaign_head_invalid
manual_implementation_hashes_missing
manual_reexecution_already_claimed_without_observed_outcome
manual_retry_deadline_spent_before_worker
manual_retry_policy_changed
manual_target_domain_invalid
manual_target_identifier_invalid
missing_reconciliation_accounting_boundary
non_campaign_archive_refused
non_campaign_tree_refused
non_json_scientific_value
noncanonical_public_trial
nonfinite_scientific_number
normalized_registered_protocol_required
observed_outcome_or_duration_invalid
observed_started_receipt_changed
operator_interruption
original_completed_result_count_changed
original_unknown_attempt_must_remain_unknown
physical_receipt_binding_changed
physical_receipt_hash_changed
physical_receipt_missing
physical_recovery_reservation_does_not_fit
physical_retry_interrupted_without_observed_outcome
physical_session_deadline_invalid
physical_session_reservation_unaffordable
prepared_split_changed
private_or_unapproved_fields
private_or_unapproved_text
published_recovery_authorization_required
published_scientific_admission_required
published_scientific_implementation_required
recovery_already_started_without_observed_outcome
recovery_authorization_binding_mismatch
recovery_authorization_bytes_required
recovery_authorization_hash_mismatch
recovery_authorization_mismatch
recovery_authorization_required
recovery_branch_ancestry_mismatch
recovery_budget_binding_mismatch
recovery_clock_invalid
recovery_clock_moved_backwards
recovery_current_repair_elapsed_required
recovery_duplicate_json_key
recovery_implementation_hash_mismatch
recovery_invalid_authorization
recovery_invalid_authorization_text
recovery_invalid_checkpoint_policy
recovery_invalid_completed_outcome
recovery_invalid_dates
recovery_invalid_hash
recovery_invalid_identity
recovery_invalid_json
recovery_invalid_lower_bound_flag
recovery_invalid_number
recovery_invalid_original_allocation
recovery_invalid_original_session
recovery_invalid_outcome
recovery_invalid_outcome_boolean
recovery_invalid_owner
recovery_invalid_phase
recovery_invalid_phase_epochs
recovery_invalid_predictions
recovery_invalid_public_outcome
recovery_invalid_reason
recovery_invalid_receipt_branch
recovery_invalid_response
recovery_invalid_run_id
recovery_invalid_target_reservation
recovery_invalid_utc
recovery_noncanonical_public_outcome
recovery_noncanonical_receipt
recovery_observed_link_mismatch
recovery_observed_lower_bound_mismatch
recovery_original_session_reused
recovery_outcome_hash_mismatch
recovery_physical_binding_mismatch
recovery_receipt_ack_unconfirmed
recovery_receipt_download_invalid
recovery_receipt_hash_mismatch
recovery_receipt_identity_mismatch
recovery_receipt_too_large
recovery_remote_binding_mismatch
recovery_request_unconfirmed
recovery_response_too_large
recovery_sealed_binding_mismatch
recovery_source_commit_invalid
recovery_started_link_mismatch
recovery_target_payload_changed
recovery_timing_mismatch
recovery_token_required
registered_device_is_cuda_0
registered_recovery_markers_missing
registered_replay_did_not_reach_recovery_target
registered_repository_required
registration_hash_mismatch
replayed_target_job_changed
restored_campaign_plan_mismatch
restored_campaign_required
result_filename_mismatch
runner_already_active
scientific_import_provenance_changed
scientific_outcome_hash_mismatch
scientific_plan_hash_mismatch
scientific_record_hash_mismatch
scientific_registration_hash_mismatch
sealed_campaign_baseline_changed_before_manual_retry
sealed_campaign_budget_changed
sealed_campaign_head_changed_before_manual_retry
sealed_campaign_head_changed_before_recovery
sealed_campaign_plan_mismatch
sealed_event_prefix_missing
sealed_old_session_allocation_changed
sealed_scientific_controls_required
sealed_started_event_changed
sealed_t4_hardware_required
selection_callback_split_changed
separate_scientific_checkout_required
snapshot_reader_cannot_write_records
source_checkout_does_not_match_pin
storage_budget_too_small
unapproved_campaign_event
unapproved_campaign_file
unapproved_pause_code
unapproved_scientific_fields
unapproved_source_hash_map
unexpected_additional_unresolved_job
unexpected_uncached_job_before_recovery
unrecognized_ledger_version
unregistered_campaign_suffix_before_retry
unregistered_python_modules
unresolved_attempt
worker_hardware_binding_changed
worker_identity_changed
worker_source_hash_changed
""".split())

@dataclass(frozen=True)
class AuthBoundIncident:
    """Incident coordinates supplied only by the independently pinned raw auth."""
    authorization: dict
    authorization_sha256: str

    @property
    def target(self):
        return self.authorization['target']

    @property
    def sealed_head(self):
        return self.authorization['sealed_campaign_head']

    @property
    def receipt_branch(self):
        return self.authorization['receipt_branch']

    @property
    def previous_completed_jobs(self):
        return self.authorization['previous_completed_jobs']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def read_json(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=pairs)
    canonical(value)
    return raw, value


def check_local_lock(directory):
    """Probe an existing advisory lock without creating or changing a file."""
    path = Path(directory) / '.campaign.lock'
    if path.is_symlink():
        raise ValueError('local_lock_path_invalid')
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except FileNotFoundError:
        return
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('runner_already_active') from None
        except OSError:
            raise ValueError('local_lock_check_unavailable') from None
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def local_diagnostics(directory, authorization=None):
    """Bounded metadata-only diagnostic; the frozen core still admits the run."""
    result = {'logical_completed_jobs': None, 'pending_job_ids': [],
              'authorized_target_completed': None, 'prior_recovery_sessions_completed': None,
              'local_snapshot_error_code': None}
    directory = Path(directory)
    if not (directory / 'campaign.json').is_file():
        return result
    def checked(path):
        if path.is_symlink() or path.stat().st_size > 8 * 1024**2:
            raise ValueError('local_metadata_invalid')
        _, value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError('local_metadata_invalid')
        content = dict(value)
        if content.pop('sha256', None) != digest(content):
            raise ValueError('local_metadata_invalid')
        return value
    try:
        configuration = checked(directory / 'campaign.json')
        if configuration.get('plan_sha256') != digest(configuration.get('plan')):
            raise ValueError('local_metadata_invalid')
        if any((directory/name).is_symlink() or not (directory/name).is_dir()
               for name in ('events', 'results')):
            raise ValueError('local_metadata_invalid')
        starts, completed, prior_recoveries = set(), set(), set()
        previous = '0' * 64
        for index, path in enumerate(sorted((directory/'events').glob('*.json'))):
            event = checked(path)
            if (path.name != '%09d.json' % index or event.get('sequence') != index
                    or event.get('previous_sha256') != previous
                    or event.get('plan_sha256') != configuration['plan_sha256']):
                raise ValueError('local_metadata_invalid')
            previous = event['sha256']
            if event.get('kind') == 'job_started':
                if not isinstance(event.get('job_id'), str) or not re.fullmatch('[0-9a-f]{64}', event['job_id']):
                    raise ValueError('local_metadata_invalid')
                starts.add(event['job_id'])
            if event.get('kind') == 'session_closed' and event.get('paused_reason') in (
                    'manual_recovery_observed_result_committed', 'conservative_recovery_session_allocation'):
                prior_recoveries.add(event['session_id'])
        for path in (directory/'results').glob('*.json'):
            value = checked(path)
            if value.get('job_id') not in starts or value['job_id'] in completed:
                raise ValueError('local_metadata_invalid')
            completed.add(value['job_id'])
        result.update(logical_completed_jobs=len(completed), pending_job_ids=sorted(starts-completed),
                      prior_recovery_sessions_completed=len(prior_recoveries))
        if authorization is not None:
            result['authorized_target_completed'] = authorization['target']['job_id'] in completed
    except (OSError, ValueError, KeyError, TypeError):
        result['local_snapshot_error_code'] = 'local_metadata_invalid'
    return result


def safe_error_code(error):
    if isinstance(error, KeyboardInterrupt):
        return 'operator_interruption'
    if type(error).__name__ == 'CampaignLocked':
        return 'runner_already_active'
    for value in (getattr(error, 'code', None), str(error)):
        if (isinstance(value, str) and (value in SAFE_ERROR_CODES
                or re.fullmatch(r'(?:github|recovery)_http_[1-5][0-9]{2}', value))):
            return value
    return 'recovery_runtime_failure'


def reserve_external_report(args, invocation_id):
    directory = Path(args.output).resolve()
    path = (Path(args.external_report) if args.external_report is not None else
            directory.parent/('recovery-v2-' + invocation_id + '.json')).resolve()
    if path == directory or directory in path.parents:
        raise ValueError('external_report_must_be_outside_campaign')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    except FileExistsError:
        raise ValueError('external_report_already_exists') from None
    return path, descriptor


def finish_external_report(reservation, value):
    path, descriptor = reservation
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()+b'\n')
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The reserved path stays evidence of this invocation; another process
        # must never replace it or reuse it for a second physical attempt.
        raise
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_authorization(path, expected_sha256, *, execute=False):
    raw, value = read_json(path)
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('authorization_hash_mismatch')
    fixed = {'version': 'e2-manual-recovery-authorization-v2',
             'repository': REPOSITORY, 'campaign_branch': BRANCH, 'campaign_prefix': BRANCH,
             'scientific_source_commit': SOURCE,
             'scientific_registration_sha256': REGISTRATION, 'plan_sha256': PLAN,
             'decision': DECISION, 'original_unknown_attempt': UNKNOWN}
    if (not isinstance(value, dict)
            or any(canonical(value.get(key)) != canonical(expected) for key, expected in fixed.items())):
        raise ValueError('authorization_binding_changed')
    target, limits = value.get('target'), value.get('limits')
    if (not isinstance(target, dict) or set(target) != {'job_id', 'job_payload_sha256',
            'started_event_sha256', 'sequence', 'old_session_id', 'reservation_seconds'}
            or not isinstance(limits, dict) or set(limits) != {'session_seconds', 'campaign_seconds',
                'old_session_conservative_allocation_seconds', 'budget_extension_authorized'}):
        raise ValueError('authorization_incident_shape_invalid')
    if (any(not isinstance(target[key], str) or not re.fullmatch('[0-9a-f]{64}', target[key])
            for key in ('job_id', 'job_payload_sha256', 'started_event_sha256'))
            or not isinstance(target['old_session_id'], str)
            or not re.fullmatch('[0-9a-f]{32}', target['old_session_id'])
            or type(target['sequence']) is not int or target['sequence'] < 1
            or type(target['reservation_seconds']) not in (int, float)
            or not math.isfinite(target['reservation_seconds'])
            or not 0 < target['reservation_seconds'] <= 28800
            or type(value.get('previous_completed_jobs')) is not int
            or value['previous_completed_jobs'] < 0
            or not isinstance(value.get('sealed_campaign_head'), str)
            or not re.fullmatch('[0-9a-f]{40}', value['sealed_campaign_head'])
            or not isinstance(value.get('receipt_branch'), str)
            or not re.fullmatch(r'recoveries/e2-cuda-selection-v1/\d{8}-r[1-9]\d*', value['receipt_branch'])
            or int(value['receipt_branch'].rsplit('-r', 1)[1]) < 2):
        raise ValueError('authorization_incident_binding_invalid')
    if (type(limits['session_seconds']) not in (int, float) or limits['session_seconds'] != 28800
            or type(limits['campaign_seconds']) not in (int, float) or limits['campaign_seconds'] != 86400
            or limits['budget_extension_authorized'] is not False
            or type(limits['old_session_conservative_allocation_seconds']) not in (int, float)
            or not math.isfinite(limits['old_session_conservative_allocation_seconds'])
            or not 0 < limits['old_session_conservative_allocation_seconds'] <= 28800):
        raise ValueError('authorization_budget_binding_invalid')
    if execute:
        if value.get('status') != 'registered_before_recovery_execution':
            raise ValueError('published_recovery_authorization_required')
        manifest = value.get('implementation_sha256')
        if (not isinstance(manifest, dict) or set(manifest) != set(CONTROLLER_FILES)
                or any(not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)
                       for sha in manifest.values())):
            raise ValueError('closed_recovery_implementation_required')
        for name, sha in manifest.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise ValueError('recovery_implementation_hash_mismatch')
    return raw, value, actual


def inspection(args):
    authorization_sha = None
    authorization_status = None
    incident = None
    if args.authorization is not None:
        _, value, authorization_sha = load_authorization(
            args.authorization, args.expected_authorization_sha256)
        authorization_status = value.get('status')
        incident = AuthBoundIncident(value, authorization_sha)
    return {'version': VERSION, 'status': 'inspection_only',
            'scientific_source_commit': SOURCE, 'scientific_registration_sha256': REGISTRATION,
            'controller_source_commit': args.controller_source_commit,
            'authorization_sha256': authorization_sha,
            'authorization_status': authorization_status, 'sealed_plan_sha256': PLAN,
            'sealed_campaign_head': None if incident is None else incident.sealed_head,
            'target': None if incident is None else dict(incident.target),
            'previous_completed_jobs': None if incident is None else incident.previous_completed_jobs,
            'session_hours': 8, 'campaign_hours': 24,
            'selection_attempts_upper_bound': 85074, 'hpo_attempts': 5544,
            'nas1_attempts': 66000, 'nas2_attempts_upper_bound': 13530,
            'receipt_branch': None if incident is None else incident.receipt_branch,
            'original_unknown_attempt': dict(UNKNOWN),
            'recovery_enabled': False, 'scientific_HPO_or_NAS_executed': False,
            'final_market_values_read': False, 'final_R': None}


def git_head(directory):
    return subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                          check=True, capture_output=True, text=True).stdout.strip()


def load_engine(engine_root):
    """Resolve every scientific import from the separate immutable checkout."""
    engine_root = Path(engine_root).resolve()
    scripts = engine_root / 'scripts'
    if engine_root == ROOT.resolve():
        raise ValueError('separate_scientific_checkout_required')
    for name, module in tuple(sys.modules.items()):
        if (name.startswith('e2_') or name.startswith('run_e2_')) and name not in (
                'e2_manual_reconciliation_v2', 'e2_recovery_receipts_v2', 'run_e2_cuda_recovery_v2'):
            origin = getattr(module, '__file__', None)
            if origin is not None and Path(origin).resolve().parent != scripts:
                raise ValueError('scientific_import_provenance_changed')
    sys.path.insert(0, str(scripts))
    names = {'frozen': 'run_e2_cuda_search', 'admission': 'e2_cuda_search_plan',
             'ledger': 'e2_search_session', 'campaign': 'e2_github_campaign',
             'transport': 'e2_github_git_transport'}
    modules = {}
    for key, name in names.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != scripts:
            raise ValueError('scientific_import_provenance_changed')
        modules[key] = module
    # New helpers are deliberately absent from the scientific checkout.
    for key, name in [('core', 'e2_manual_reconciliation_v2'), ('receipts', 'e2_recovery_receipts_v2')]:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != ROOT / 'scripts':
            raise ValueError('controller_import_provenance_changed')
        modules[key] = module
    return SimpleNamespace(**modules)


def preflight(args, *, phase_callback=None, authorization_callback=None):
    mark = phase_callback or (lambda phase: None)
    mark('authorization_validation')
    if not args.retry_lost_once:
        raise ValueError('explicit_retry_lost_once_required')
    if (args.source_commit != SOURCE or args.expected_registration_sha256 != REGISTRATION
            or args.github_repository != REPOSITORY or args.github_branch != BRANCH
            or args.github_prefix != BRANCH or args.device != 'cuda:0'
            or args.session_hours != 8 or args.campaign_hours != 24
            or args.storage_gib != 10 or args.extend_campaign_hours is not None):
        raise ValueError('sealed_scientific_controls_required')
    if not re.fullmatch('[0-9a-f]{40}', args.controller_source_commit or ''):
        raise ValueError('immutable_controller_commit_required')
    raw, authorization, sha = load_authorization(
        args.authorization, args.expected_authorization_sha256, execute=True)
    if authorization_callback is not None:
        authorization_callback(authorization, sha)
    mark('controller_source_validation')
    if git_head(ROOT) != args.controller_source_commit:
        raise ValueError('controller_checkout_does_not_match_pin')
    engine = load_engine(args.engine_root)
    mark('scientific_source_validation')
    engine.frozen.verify_source(SOURCE)
    registration_raw, registered = read_json(engine.admission.REGISTRATION)
    if hashlib.sha256(registration_raw).hexdigest() != REGISTRATION:
        raise ValueError('scientific_registration_hash_mismatch')
    if (registered.get('status') != 'registered_before_scientific_cuda_fits'
            or not registered.get('implementation_sha256')
            or engine.admission.source_hashes() != registered['implementation_sha256']):
        raise ValueError('published_scientific_implementation_required')
    # A quota-exhausted or unavailable T4 consumes no one-use receipt and appends
    # no scientific ledger event. No mirror or receipt store exists yet.
    mark('cuda_preflight')
    hardware = engine.frozen.cuda_hardware()
    if hardware.get('device_name') != 'Tesla T4':
        raise ValueError('sealed_t4_hardware_required')
    mark('sealed_plan_validation')
    plan = engine.admission.build_plan(source_commit=SOURCE, models=args.models, origins=args.origins,
        session_hours=8, campaign_hours=24, storage_gib=10, device='cuda:0', hardware=hardware,
        expected_registration_sha256=REGISTRATION, github_branch=BRANCH, github_prefix=BRANCH)
    if digest(plan) != PLAN:
        raise ValueError('sealed_campaign_plan_mismatch')
    return engine, raw, authorization, sha, plan


class RecoveryStopped(Exception):
    def __init__(self, result):
        self.result = result
        super().__init__('target_recovery_finished')


class ReplayFacade:
    """Replay the genuine cached outcomes and stop at the authorized target."""
    def __init__(self, view, authorization, reconciler):
        self.view = view
        self.authorization = authorization
        self.reconciler = reconciler
        self.invocations = 0

    def get_result(self, job_id, payload=None):
        value = self.view.get_result(job_id, payload)
        # Capture the exact target payload even when a prior OBSERVED receipt
        # already repaired it; the reconciler then validates without refitting.
        return None if job_id == self.authorization['target']['job_id'] else value

    def run_job(self, job_id, payload, callback, *, reservation_seconds):
        target = self.authorization['target']
        if job_id != target['job_id']:
            cached = self.view.get_result(job_id, payload)
            if cached is None:
                raise ValueError('unexpected_uncached_job_before_recovery')
            return cached
        if (digest(payload) != target['job_payload_sha256']
                or digest({'plan_sha256': self.view.plan_sha256, 'job': payload}) != job_id
                or reservation_seconds != target['reservation_seconds'] or self.invocations):
            raise ValueError('recovery_target_payload_changed')
        self.invocations += 1
        result = self.reconciler.recover(job_id, payload, callback,
                                         reservation_seconds=reservation_seconds)
        raise RecoveryStopped(result)


class DeadlineSession:
    """Check the physical runtime deadline before a new durable STARTED event."""
    def __init__(self, session, deadline, *, clock=time.monotonic):
        self.session, self.deadline, self.clock = session, deadline, clock

    def __getattr__(self, key):
        return getattr(self.session, key)

    def run_job(self, job_id, payload, callback, *, reservation_seconds):
        if self.session.get_result(job_id, payload) is None:
            remaining = self.deadline - self.clock()
            if reservation_seconds > remaining:
                self.session._pause('session_budget', reservation_seconds=reservation_seconds,
                                    remaining_seconds=max(0.0, remaining))
        return self.session.run_job(job_id, payload,
            lambda deadline: callback(min(deadline, self.deadline)), reservation_seconds=reservation_seconds)


class HandoffClock:
    """Begin ledger accounting at an already observed reconciliation boundary.

    Frozen SearchSession first reads its injected clock in _start_session.
    Subsequent reads use the live monotonic clock. No scientific source or
    ledger timers are modified; the external recovery result supplies the
    earlier boundary so final reconciliation ACKs are charged exactly once.
    """
    def __init__(self, accounted_through, clock):
        self.accounted_through, self.clock, self.first = accounted_through, clock, True

    def __call__(self):
        if self.first:
            self.first = False
            return self.accounted_through
        return self.clock()


def continue_selection(args, engine, plan, directory, mirror, panel, deadline, *, clock=time.monotonic,
                       repair_only=False, accounted_through=None, context=None):
    """Use the frozen selection and real ledger with the remaining physical cap."""
    session = None
    status, exit_code, selections = 'not_started', 0, []
    pause_reason = retry_after = None
    def failure(error):
        if context is not None:
            context['error_code'] = safe_error_code(error)
            context['error_phase'] = context['phase']
    now = clock()
    accounted_through = now if accounted_through is None else accounted_through
    if (type(accounted_through) not in (int, float) or not math.isfinite(accounted_through)
            or accounted_through > now):
        raise ValueError('invalid_reconciliation_accounting_boundary')
    remaining = max(0.0, deadline - accounted_through)
    accounting_clock = HandoffClock(accounted_through, clock)
    try:
        # A tiny positive cap allows a metadata-only final report after expiry;
        # DeadlineSession still refuses every uncached scientific attempt.
        session = engine.ledger.SearchSession(directory, plan,
            session_budget_seconds=max(remaining, 0.000001),
            campaign_budget_seconds=86400, max_storage_bytes=10 * 1024**3, clock=accounting_clock)
        session._start_session()  # Cached replay and remote sync consume this cap.
        paths = session.pending_record_paths
        mirror.push(directory, paths=paths, message='Resume frozen CUDA selection after manual reconciliation')
        session.mark_records_synced(paths)
        if repair_only:
            status = 'interrupted_between_jobs'
            pause_reason = 'manual_recovery_completed_resume_registered_selection'
        else:
            wrapped = DeadlineSession(session, deadline, clock=clock)
            def worker(job, worker_deadline):
                if context is not None:
                    context['phase'] = 'normal_physical_worker'
                value = engine.frozen.isolated_trial(plan, job, data_root=args.data_root,
                    source_root_itransformer=args.source_root_itransformer,
                    deadline=min(worker_deadline, deadline))
                if context is not None:
                    context['phase'] = 'normal_continuation'
                return value
            engine.frozen.run_selection(plan, wrapped, panel, worker=worker, mirror=mirror, progress=selections)
            status = 'completed_selected_registered_domains'
    except engine.ledger.SessionPaused as error:
        status, pause_reason = 'paused_' + error.reason, error.reason
    except engine.ledger.UnresolvedAttempt as error:
        cancelled = getattr(error, 'details', {}).get('exception_type') == 'KeyboardInterrupt'
        status, exit_code = ('interrupted_between_jobs', 130) if cancelled else ('reconciliation_required', 3)
        if context is not None:
            context.update(error_code='operator_interruption' if cancelled else 'unresolved_attempt',
                           error_phase=context['phase'])
    except engine.campaign.CampaignSyncError as error:
        failure(error)
        status, exit_code, pause_reason = 'paused_github_sync', 4, error.code
        retry_after = getattr(error, 'retry_after_seconds', None)
    except KeyboardInterrupt as error:
        failure(error)
        status, pause_reason, exit_code = 'interrupted_between_jobs', 'operator_interruption', 130
    except Exception as error:
        failure(error)
        # Import only after hardware admission; this is the frozen typed policy.
        temporal = importlib.import_module('e2_temporal_training')
        if isinstance(error, temporal.SelectionAborted):
            status, exit_code = 'blocked_selection_failure', 5
        else:
            status, pause_reason, exit_code = 'blocked_runtime_failure', 'runtime_failure', 2
    finally:
        if session is not None:
            session.close()
            state = engine.frozen.report(plan, session, status, selections, REPOSITORY, BRANCH,
                                         pause_reason=pause_reason, retry_after_seconds=retry_after)
            engine.frozen.write_json(directory / 'status.json', state)
            try:
                if context is not None:
                    context['phase'] = 'continuation_publication'
                mirror.push(directory, paths=(*session.pending_record_paths, 'status.json'),
                            message='Save frozen selection pause or completion after manual recovery')
            except engine.campaign.CampaignSyncError as error:
                failure(error)
                status, exit_code = 'paused_github_sync', 4
                state.update(status=status, pause_reason=pause_reason or error.code,
                    retry_after_seconds=retry_after if retry_after is not None else getattr(error, 'retry_after_seconds', None))
                engine.frozen.write_json(directory / 'status.json', state)
    return exit_code, status, None if session is None else len(session.completed_job_ids)


def external_report(args, context, started, *, clock=time.monotonic):
    elapsed = max(0.0, clock() - started)
    authorization = context.get('authorization')
    recovery_result = context.get('recovery_result', {})
    summary_keys = ('status', 'job_id', 'recovery_session_id', 'actual_fit_wall_seconds',
                    'recovery_elapsed_seconds', 'session_seconds_remaining',
                    'campaign_charged_seconds', 'completed_jobs',
                    'physical_reexecution_performed', 'campaign_commit',
                    'campaign_accounted_through_monotonic', 'continuation_session_budget_seconds')
    diagnostics = context.get('diagnostics', {})
    return {'version': VERSION, 'invocation_id': context['invocation_id'],
            'status': context['status'], 'phase': context['phase'],
            'error_code': context.get('error_code'), 'error_phase': context.get('error_phase'),
            'report_persistence_error_code': context.get('report_persistence_error_code'),
            'scientific_source_commit': SOURCE,
            'scientific_registration_sha256': REGISTRATION,
            'controller_source_commit': args.controller_source_commit,
            'authorization_sha256': context.get('authorization_sha256'), 'sealed_plan_sha256': PLAN,
            'sealed_campaign_head': None if authorization is None else authorization['sealed_campaign_head'],
            'authorized_target': None if authorization is None else dict(authorization['target']),
            'logical_completed_jobs': diagnostics.get('logical_completed_jobs'),
            'pending_job_ids': diagnostics.get('pending_job_ids', []),
            'authorized_target_completed': diagnostics.get('authorized_target_completed'),
            'prior_recovery_sessions_completed': diagnostics.get('prior_recovery_sessions_completed'),
            'local_snapshot_error_code': diagnostics.get('local_snapshot_error_code'),
            'recovery_result': {key: recovery_result[key] for key in summary_keys if key in recovery_result},
            'physical_worker_started': context.get('physical_worker_started', False),
            'physical_worker_returned': context.get('physical_worker_returned', False),
            'original_unknown_attempt': dict(UNKNOWN),
            'physical_session_elapsed_seconds': elapsed,
            'physical_session_remaining_seconds': max(0.0, 28800 - elapsed),
            'physical_session_limit_seconds': 28800,
            'physical_session_limit_overrun_seconds': max(0.0, elapsed - 28800),
            'receipt_repository': REPOSITORY,
            'receipt_branch': None if authorization is None else authorization['receipt_branch'],
            'original_physical_attempt_still_unknown': True,
            'campaign_accounting_policy': 'Measured reconciliation ACKs and cached replay are charged through the continuation ledger closure. Publication after that final closure follows the frozen boundary policy; the external physical elapsed time includes that publication tail.',
            'final_market_values_read': False, 'final_R': None}


def execute(args, *, clock=time.monotonic):
    # Physical8h includes checks, restore, all ACKs, deterministic replay and fits.
    started = clock()
    deadline = started + 28800
    context = {'invocation_id': uuid.uuid4().hex, 'status': 'not_started',
               'phase': 'local_lock_check', 'error_code': None, 'error_phase': None,
               'recovery_result': {}, 'diagnostics': {}}
    reservation = mirror = None
    exit_code = 2
    directory = Path(args.output)
    def phase(value):
        context['phase'] = value
    try:
        # No source import, GPU query, network operation or write precedes this.
        check_local_lock(directory)
        phase('report_reservation')
        reservation = reserve_external_report(args, context['invocation_id'])
        context['diagnostics'] = local_diagnostics(directory)
        engine, raw, authorization, authorization_sha, plan = preflight(args, phase_callback=phase,
            authorization_callback=lambda authorization, sha: context.update(
                authorization=authorization, authorization_sha256=sha))
        context.update(authorization=authorization, authorization_sha256=authorization_sha)
        incident = AuthBoundIncident(authorization, authorization_sha)
        context['diagnostics'] = local_diagnostics(directory, authorization)
        phase('github_credentials')
        if not os.environ.get('GITHUB_TOKEN'):
            raise ValueError('github_secret_required')
        phase('campaign_restore')
        mirror = engine.transport.GitCampaignMirror(REPOSITORY, BRANCH, BRANCH,
                                                    max_snapshot_bytes=2 * 1024**3)
        restored = mirror.restore(directory)
        context['diagnostics'] = local_diagnostics(directory, authorization)
        phase('recovery_receipt_inspection')
        receipts = engine.receipts.RecoveryReceiptStore(REPOSITORY, incident.receipt_branch,
            authorization_sha, incident.sealed_head, authorization=authorization, authorization_bytes=raw)
        receipt_state = receipts.inspect()
        if receipt_state.get('started') and not receipt_state.get('observed'):
            raise ValueError('recovery_already_started_without_observed_outcome')
        if not receipt_state.get('started') and restored.get('remote_commit') != incident.sealed_head:
            raise ValueError('sealed_campaign_head_changed_before_recovery')
        phase('restored_plan_validation')
        _, configuration = read_json(directory/'campaign.json')
        if (not isinstance(configuration, dict) or configuration.get('plan') != plan
                or configuration.get('plan_sha256') != PLAN or restored.get('plan_sha256') != PLAN):
            raise ValueError('restored_campaign_plan_mismatch')
        phase('snapshot_validation')
        view = engine.core.read_snapshot(directory, plan, authorization,
                                         session_class=engine.ledger.SearchSession)
        reconciler = engine.core.ManualReconciler(view, authorization, mirror, receipts,
            authorization_sha256=authorization_sha, physical_session_started_at=started,
            physical_deadline=deadline, recovery_source_commit=args.controller_source_commit, clock=clock)
        if deadline-clock() < incident.target['reservation_seconds'] and not receipt_state.get('observed'):
            raise ValueError('physical_recovery_reservation_does_not_fit')
        phase('development_preparation')
        panel, _ = engine.frozen.prepare_panel(args.data_root)
        phase('deterministic_replay')
        replay = ReplayFacade(view, authorization, reconciler)
        def worker(job, worker_deadline):
            phase('recovery_physical_worker')
            context['physical_worker_started'] = True
            value = engine.frozen.isolated_trial(plan, job, data_root=args.data_root,
                source_root_itransformer=args.source_root_itransformer,
                deadline=min(worker_deadline, deadline))
            context['physical_worker_returned'] = True
            phase('recovery_observation_and_commit')
            return value
        try:
            engine.frozen.run_selection(plan, replay, panel, worker=worker)
        except RecoveryStopped as stopped:
            context['recovery_result'] = stopped.result
        else:
            raise ValueError('registered_replay_did_not_reach_recovery_target')
        accounted_through = context['recovery_result'].get('campaign_accounted_through_monotonic')
        if (type(accounted_through) not in (int, float) or not math.isfinite(accounted_through)
                or not started <= accounted_through <= clock()):
            raise ValueError('missing_reconciliation_accounting_boundary')
        phase('normal_continuation')
        exit_code, status, completed = continue_selection(args, engine, plan, directory,
            mirror, panel, deadline, clock=clock, repair_only=args.repair_only,
            accounted_through=accounted_through, context=context)
        context['status'] = status
        if completed is not None:
            context['diagnostics']['logical_completed_jobs'] = completed
    except (Exception, KeyboardInterrupt) as error:
        code = safe_error_code(error)
        context.update(error_code=code, error_phase=context['phase'])
        if code == 'runner_already_active':
            context['status'] = 'runner_already_active'
        elif code in ('unexpected_additional_unresolved_job', 'unresolved_attempt'):
            context['status'] = 'blocked_new_unresolved_job'
        else:
            context['status'] = 'blocked_preflight' if mirror is None else 'blocked_recovery'
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        if mirror is not None:
            try:
                mirror.close()
            except Exception as error:
                if context['error_code'] is None:
                    context.update(status='blocked_runtime_failure', error_code=safe_error_code(error),
                                   error_phase='transport_close')
                    exit_code = 2
        diagnostics = local_diagnostics(directory, context.get('authorization'))
        if diagnostics['logical_completed_jobs'] is not None or not context['diagnostics']:
            context['diagnostics'] = diagnostics
        if reservation is None and context['error_code'] not in (
                'external_report_already_exists', 'external_report_must_be_outside_campaign'):
            try:
                reservation = reserve_external_report(args, context['invocation_id'])
            except Exception as error:
                context['report_persistence_error_code'] = safe_error_code(error)
        destination = None
        state = external_report(args, context, started, clock=clock)
        if reservation is not None:
            try:
                destination = finish_external_report(reservation, state)
            except Exception:
                context.update(status='blocked_runtime_failure', error_code='external_report_persistence_failed',
                    error_phase='external_report_publication', report_persistence_error_code='external_report_persistence_failed')
                exit_code = 2
        print(json.dumps({'status': context['status'], 'error_code': context['error_code'],
            'error_phase': context['error_phase'], 'logical_completed_jobs': state['logical_completed_jobs'],
            'pending_job_ids': state['pending_job_ids'],
            'external_report': None if destination is None else str(destination),
            'report_persistence_error_code': context.get('report_persistence_error_code'),
            'receipt_branch': state['receipt_branch'],
            'original_physical_attempt_still_unknown': True,
            'final_market_values_read': False, 'final_R': None}, ensure_ascii=False))
    return exit_code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only', '--inspect', dest='plan_only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    result.add_argument('--engine-root', type=Path)
    result.add_argument('--controller-source-commit')
    result.add_argument('--authorization', type=Path)
    result.add_argument('--expected-authorization-sha256')
    result.add_argument('--retry-lost-once', action='store_true')
    result.add_argument('--repair-only', action='store_true')
    result.add_argument('--source-commit', default=SOURCE)
    result.add_argument('--expected-registration-sha256', default=REGISTRATION)
    result.add_argument('--output', type=Path)
    result.add_argument('--external-report', type=Path)
    result.add_argument('--data-root', type=Path)
    result.add_argument('--source-root-itransformer')
    result.add_argument('--models', nargs='+')
    result.add_argument('--origins', nargs='+')
    result.add_argument('--cohort', choices=['large'], default='large')
    result.add_argument('--session-hours', type=float, default=8)
    result.add_argument('--campaign-hours', type=float, default=24)
    result.add_argument('--storage-gib', type=float, default=10)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--extend-campaign-hours', type=float)
    result.add_argument('--github-repository', default=REPOSITORY)
    result.add_argument('--github-branch', default=BRANCH)
    result.add_argument('--github-prefix', default=BRANCH)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.plan_only:
        print(json.dumps(inspection(args), ensure_ascii=False, indent=2))
        return 0
    required = ('engine_root', 'controller_source_commit', 'authorization',
                'expected_authorization_sha256', 'output', 'data_root')
    if any(getattr(args, key) is None for key in required):
        cli.error('execution requires engine/controller/authentication pins, --output and --data-root')
    return execute(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        code = str(error) if re.fullmatch('[a-z][a-z0-9_]{1,95}', str(error)) else 'recovery_preflight_or_runtime_failed'
        print(json.dumps({'status': 'blocked_recovery', 'error_code': code,
                          'automatic_retry_permitted': False}), file=sys.stderr)
        raise SystemExit(2)
