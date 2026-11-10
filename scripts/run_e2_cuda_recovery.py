"""One explicitly authorized external reconciliation of the sealed A3 campaign.

Inspection uses only the standard library. The scientific checkout, admission,
plan, data policy and isolated worker remain unchanged. A receipt on a separate
branch records the additional physical attempt; its predecessor stays unknown.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'e2-external-cuda-recovery-v1'
REPOSITORY = 'jassan-bit/DeepLearning_202630'
BRANCH = 'campaigns/e2-cuda-selection-v1'
RECEIPT_BRANCH = 'recoveries/e2-cuda-selection-v1/20261110'
SOURCE = '564ba4762f84a800b02fa4c734f5752a169f5223'
REGISTRATION = '78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9'
PLAN = '4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda'
SEALED_HEAD = '5648b8305174a827d1ffa0a90ce58dab1ad5b71c'
TARGET = {
    'job_id': '3775f65fd049666c641c1131e94726e12e98143f91a755075c07ac6761dd668b',
    'job_payload_sha256': '0eb5d72e23cb399edb41275467b3ae05c6cc856999f192c4cb254d25de49bf34',
    'started_event_sha256': '5afba2cd98646e734799c0ae7ef3bdbc86d4688eda63648354d8a2c757ead1f5',
    'sequence': 989, 'old_session_id': '9bc856c8bdb34536ad8f01de065e2261',
    'reservation_seconds': 5700,
}
LIMITS = {'session_seconds': 28800, 'campaign_seconds': 86400,
          'old_session_conservative_allocation_seconds': 28800,
          'budget_extension_authorized': False}
UNKNOWN = {'outcome': 'unknown', 'elapsed_seconds': None, 'optimizer_steps': None,
           'optimizer_steps_known_lower_bound': 0,
           'partial_optimizer_update_may_have_occurred': True}
CONTROLLER_FILES = ('scripts/run_e2_cuda_recovery.py',
                    'scripts/e2_manual_reconciliation.py',
                    'scripts/e2_recovery_receipts.py')


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
    fixed = {'version': 'e2-manual-recovery-authorization-v1',
             'repository': REPOSITORY, 'campaign_branch': BRANCH, 'campaign_prefix': BRANCH,
             'receipt_branch': RECEIPT_BRANCH, 'scientific_source_commit': SOURCE,
             'scientific_registration_sha256': REGISTRATION, 'plan_sha256': PLAN,
             'sealed_campaign_head': SEALED_HEAD, 'previous_completed_jobs': 494,
             'target': TARGET, 'limits': LIMITS, 'original_unknown_attempt': UNKNOWN}
    if (not isinstance(value, dict)
            or any(canonical(value.get(key)) != canonical(expected) for key, expected in fixed.items())):
        raise ValueError('authorization_binding_changed')
    # bool/int equality is insufficient for the immutable operational limits.
    if (type(value['previous_completed_jobs']) is not int
            or type(value['limits']['budget_extension_authorized']) is not bool
            or any(type(value['limits'][key]) is not int for key in LIMITS
                   if key != 'budget_extension_authorized')
            or any(type(value['target'][key]) is not int
                   for key in ('sequence', 'reservation_seconds'))):
        raise ValueError('authorization_binding_type_changed')
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
    if args.authorization is not None:
        _, value, authorization_sha = load_authorization(
            args.authorization, args.expected_authorization_sha256)
        authorization_status = value.get('status')
    return {'version': VERSION, 'status': 'inspection_only',
            'scientific_source_commit': SOURCE, 'scientific_registration_sha256': REGISTRATION,
            'controller_source_commit': args.controller_source_commit,
            'authorization_sha256': authorization_sha,
            'authorization_status': authorization_status, 'sealed_plan_sha256': PLAN,
            'sealed_campaign_head': SEALED_HEAD, 'target': dict(TARGET),
            'previous_completed_jobs': 494, 'session_hours': 8, 'campaign_hours': 24,
            'selection_attempts_upper_bound': 85074, 'hpo_attempts': 5544,
            'nas1_attempts': 66000, 'nas2_attempts_upper_bound': 13530,
            'receipt_branch': RECEIPT_BRANCH, 'original_unknown_attempt': dict(UNKNOWN),
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
                'e2_manual_reconciliation', 'e2_recovery_receipts', 'run_e2_cuda_recovery'):
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
    for key, name in [('core', 'e2_manual_reconciliation'), ('receipts', 'e2_recovery_receipts')]:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != ROOT / 'scripts':
            raise ValueError('controller_import_provenance_changed')
        modules[key] = module
    return SimpleNamespace(**modules)


def preflight(args):
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
    if git_head(ROOT) != args.controller_source_commit:
        raise ValueError('controller_checkout_does_not_match_pin')
    engine = load_engine(args.engine_root)
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
    hardware = engine.frozen.cuda_hardware()
    if hardware.get('device_name') != 'Tesla T4':
        raise ValueError('sealed_t4_hardware_required')
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
    """Replay the 494 genuine cached outcomes and stop at the exact target."""
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
                       repair_only=False, accounted_through=None):
    """Use the frozen selection and real ledger with the remaining physical cap."""
    session = None
    status, exit_code, selections = 'not_started', 0, []
    pause_reason = retry_after = None
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
                return engine.frozen.isolated_trial(plan, job, data_root=args.data_root,
                    source_root_itransformer=args.source_root_itransformer,
                    deadline=min(worker_deadline, deadline))
            engine.frozen.run_selection(plan, wrapped, panel, worker=worker, mirror=mirror, progress=selections)
            status = 'completed_selected_registered_domains'
    except engine.ledger.SessionPaused as error:
        status, pause_reason = 'paused_' + error.reason, error.reason
    except engine.ledger.UnresolvedAttempt:
        status, exit_code = 'reconciliation_required', 3
    except engine.campaign.CampaignSyncError as error:
        status, exit_code, pause_reason = 'paused_github_sync', 4, error.code
        retry_after = getattr(error, 'retry_after_seconds', None)
    except KeyboardInterrupt:
        status, pause_reason, exit_code = 'interrupted_between_jobs', 'operator_interruption', 130
    except Exception as error:
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
                mirror.push(directory, paths=(*session.pending_record_paths, 'status.json'),
                            message='Save frozen selection pause or completion after manual recovery')
            except engine.campaign.CampaignSyncError as error:
                status, exit_code = 'paused_github_sync', 4
                state.update(status=status, pause_reason=pause_reason or error.code,
                    retry_after_seconds=retry_after if retry_after is not None else getattr(error, 'retry_after_seconds', None))
                engine.frozen.write_json(directory / 'status.json', state)
    return exit_code, status, None if session is None else len(session.completed_job_ids)


def external_report(args, authorization_sha, recovery_result, status, completed, started, *, clock=time.monotonic):
    elapsed = max(0.0, clock() - started)
    # Physical quantities from a fresh, independently observed retry never
    # replace unknown quantities from the lost original physical execution.
    summary_keys = ('status', 'job_id', 'recovery_session_id', 'actual_fit_wall_seconds',
                    'recovery_elapsed_seconds', 'session_seconds_remaining',
                    'campaign_charged_seconds', 'completed_jobs',
                    'physical_reexecution_performed', 'campaign_commit',
                    'campaign_accounted_through_monotonic', 'continuation_session_budget_seconds')
    return {'version': VERSION, 'status': status, 'scientific_source_commit': SOURCE,
            'controller_source_commit': args.controller_source_commit,
            'authorization_sha256': authorization_sha, 'sealed_plan_sha256': PLAN,
            'logical_completed_jobs': completed,
            'recovery_result': {key: recovery_result[key] for key in summary_keys if key in recovery_result},
            'original_unknown_attempt': dict(UNKNOWN),
            'physical_session_elapsed_seconds': elapsed,
            'physical_session_remaining_seconds': max(0.0, 28800 - elapsed),
            'physical_session_limit_seconds': 28800,
            'physical_session_limit_overrun_seconds': max(0.0, elapsed - 28800),
            'receipt_repository': REPOSITORY, 'receipt_branch': RECEIPT_BRANCH,
            'original_physical_attempt_still_unknown': True,
            'campaign_accounting_policy': 'Measured reconciliation ACKs and cached replay are charged through the continuation ledger closure. Publication after that final closure follows the frozen boundary policy; the external physical elapsed time includes that publication tail.',
            'final_market_values_read': False, 'final_R': None}


def execute(args, *, clock=time.monotonic):
    # Includes source checks, hardware probing, restore, replay and remote ACKs.
    started = clock()
    deadline = started + 28800
    engine, raw, authorization, authorization_sha, plan = preflight(args)
    if not os.environ.get('GITHUB_TOKEN'):
        raise ValueError('github_secret_required')
    directory = Path(args.output)
    mirror = engine.transport.GitCampaignMirror(REPOSITORY, BRANCH, BRANCH,
                                                max_snapshot_bytes=2 * 1024**3)
    recovery_result, status, completed, exit_code = {}, 'reconciliation_required', None, 3
    try:
        restored = mirror.restore(directory)
        # On the first physical claim the campaign must still be the exact head
        # the operator reviewed; later receipt validation handles idempotency.
        receipts = engine.receipts.RecoveryReceiptStore(REPOSITORY, RECEIPT_BRANCH,
            authorization_sha, SEALED_HEAD, authorization=authorization, authorization_bytes=raw)
        receipt_state = receipts.inspect()
        if receipt_state.get('started') and not receipt_state.get('observed'):
            raise ValueError('recovery_already_started_without_observed_outcome')
        if not receipt_state.get('started') and restored.get('remote_commit') != SEALED_HEAD:
            raise ValueError('sealed_campaign_head_changed_before_recovery')
        _, configuration = read_json(directory / 'campaign.json')
        if (not isinstance(configuration, dict) or configuration.get('plan') != plan or configuration.get('plan_sha256') != PLAN
                or restored.get('plan_sha256') != PLAN):
            raise ValueError('restored_campaign_plan_mismatch')
        view = engine.core.read_snapshot(directory, plan, authorization,
                                         session_class=engine.ledger.SearchSession)
        reconciler = engine.core.ManualReconciler(view, authorization, mirror, receipts,
            authorization_sha256=authorization_sha, physical_session_started_at=started,
            physical_deadline=deadline, recovery_source_commit=args.controller_source_commit, clock=clock)
        if deadline - clock() < TARGET['reservation_seconds'] and not receipt_state.get('observed'):
            raise ValueError('physical_recovery_reservation_does_not_fit')
        panel, _ = engine.frozen.prepare_panel(args.data_root)
        replay = ReplayFacade(view, authorization, reconciler)
        def worker(job, worker_deadline):
            return engine.frozen.isolated_trial(plan, job, data_root=args.data_root,
                source_root_itransformer=args.source_root_itransformer,
                deadline=min(worker_deadline, deadline))
        try:
            engine.frozen.run_selection(plan, replay, panel, worker=worker)
        except RecoveryStopped as stopped:
            recovery_result = stopped.result
        else:
            raise ValueError('registered_replay_did_not_reach_recovery_target')
        accounted_through = recovery_result.get('campaign_accounted_through_monotonic')
        if (type(accounted_through) not in (int, float) or not math.isfinite(accounted_through)
                or not started <= accounted_through <= clock()):
            raise ValueError('missing_reconciliation_accounting_boundary')
        exit_code, status, completed = continue_selection(args, engine, plan, directory,
            mirror, panel, deadline, clock=clock, repair_only=args.repair_only,
            accounted_through=accounted_through)
    finally:
        # An external local report never enters the frozen campaign manifest.
        state = external_report(args, authorization_sha, recovery_result, status, completed, started, clock=clock)
        destination = directory.parent / ('recovery-' + authorization_sha[:16] + '.json')
        try:
            write_json(destination, state)
        finally:
            mirror.close()
        print(json.dumps({'status': status, 'logical_completed_jobs': completed,
            'external_report': destination.name, 'receipt_branch': RECEIPT_BRANCH,
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
