"""Recover one registered interrupted optimized E2 job, then continue unchanged selection.

No retry or budget amendment is inferred from a diagnostic. Execute requires an
explicit combined total, registered durably before the one physical retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "564ba4762f84a800b02fa4c734f5752a169f5223"
OPTIMIZATION = "8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9"
REPOSITORY = "jassan-bit/DeepLearning_202630"
BRANCH = "campaigns/e2-cuda-crop-11-v2"
SEALED_HEAD = "30c121072f97917c234fea2269dad50d00113850"
PLAN = "877a8ab4a0d34599f43c5dd16a90fe87fe03d37fe535466bf8e009c379cce742"
REGISTRATION = ROOT / "experiments/e2_optimized_recovery_20261112/registration.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def checked_source(directory, expected):
    directory = Path(directory).resolve()
    actual = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected:
        raise ValueError("recovery_source_checkout_mismatch")
    changed = subprocess.run(["git", "-C", str(directory), "diff", "--name-only", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    extras = subprocess.run(["git", "-C", str(directory), "ls-files", "--others",
                             "--exclude-standard", "scripts"], check=True,
                            capture_output=True, text=True).stdout.splitlines()
    if changed or any(p.endswith(".py") for p in extras):
        raise ValueError("recovery_source_has_unregistered_changes")


def assert_no_other_executor():
    markers = ("run_e2_cuda_search.py", "run_e2_cuda_recovery_",
               "e2_guide_schedule.py", "run_e2_optimized_selection",
               "run_e2_optimized_recovery_v1.py")
    output = subprocess.run(["ps", "-eo", "pid=,args="], check=True,
                            capture_output=True, text=True).stdout
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0] != str(os.getpid()) and any(m in fields[1] for m in markers):
            raise ValueError("another_executor_is_active")


def load_registered(args):
    checked_source(ROOT, args.recovery_commit)
    registered = json.loads(REGISTRATION.read_bytes())
    if (registered.get("status") != "registered_before_any_recovery_execution"
            or registered.get("sealed_campaign_head") != SEALED_HEAD
            or registered.get("plan_sha256") != PLAN):
        raise ValueError("recovery_registration_mismatch")
    for relative, expected in registered["implementation_sha256"].items():
        path = ROOT / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("recovery_implementation_changed")
    checked_source(args.optimization_root, OPTIMIZATION)
    sys.path.insert(0, str(Path(args.optimization_root).resolve() / "scripts"))
    optimized = importlib.import_module("run_e2_optimized_selection_v2")
    if Path(optimized.__file__).resolve().parent != Path(args.optimization_root).resolve() / "scripts":
        raise ValueError("optimized_import_provenance_changed")
    optimized.read_registration()
    runner = optimized.load_engine(args.engine_root)
    import e2_github_campaign as public
    optimized.install_public_schema(public)
    import e2_search_session as ledger
    from e2_github_git_transport import GitCampaignMirror
    import e2_optimized_reconciliation_v1 as core
    from e2_optimized_recovery_git_v1 import RecoveryGitTransport
    return registered, optimized, runner, public, ledger, GitCampaignMirror, core, RecoveryGitTransport


class RecoveryStopped(Exception):
    def __init__(self, result):
        self.result = result


class ReplayFacade:
    """Replay cached outcomes and intercept exactly the sealed missing job."""
    def __init__(self, view, authorization, reconciler):
        self.view, self.authorization, self.reconciler = view, authorization, reconciler
        self.directory = view.directory
        self.invocations = 0

    @property
    def pending_record_paths(self):
        return ()

    def mark_records_synced(self, paths):
        if paths:
            raise ValueError("replay_cannot_publish_other_records")

    def get_result(self, job_id, payload=None):
        result = self.view.get_result(job_id, payload)
        return None if job_id == self.authorization["target"]["job_id"] else result

    def run_job(self, job_id, payload, callback, *, reservation_seconds):
        target = self.authorization["target"]
        if job_id != target["job_id"]:
            cached = self.view.get_result(job_id, payload)
            if cached is None:
                raise ValueError("unexpected_uncached_job_before_target")
            return cached
        if (digest(payload) != target["job_payload_sha256"]
                or digest({"plan_sha256": self.view.plan_sha256, "job": payload}) != job_id
                or reservation_seconds != target["reservation_seconds"] or self.invocations):
            raise ValueError("recovery_target_payload_changed")
        self.invocations += 1
        result = self.reconciler.recover(job_id, payload, callback,
                                         reservation_seconds=reservation_seconds)
        raise RecoveryStopped(result)


class HandoffClock:
    def __init__(self, point, clock):
        self.point, self.clock, self.first = point, clock, True

    def __call__(self):
        if self.first:
            self.first = False
            return self.point
        return self.clock()


class DeadlineSession:
    def __init__(self, session, deadline, clock):
        self.session, self.deadline, self.clock = session, deadline, clock

    def __getattr__(self, key):
        return getattr(self.session, key)

    def run_job(self, job_id, payload, callback, *, reservation_seconds):
        if self.session.get_result(job_id, payload) is None:
            remaining = self.deadline - self.clock()
            if reservation_seconds > remaining:
                self.session._pause("session_budget", reservation_seconds=reservation_seconds,
                                    remaining_seconds=max(0.0, remaining))
        return self.session.run_job(job_id, payload,
            lambda deadline: callback(min(deadline, self.deadline)),
            reservation_seconds=reservation_seconds)


def continue_selection(args, optimized, runner, ledger, plan, legacy_plan, legacy,
                       mirror, panel, deadline, accounted_through, *, clock=time.monotonic,
                       on_new_fit=None):
    now = clock()
    if not isinstance(accounted_through, (int, float)) or not math.isfinite(accounted_through) or accounted_through > now:
        raise ValueError("missing_reconciliation_accounting_boundary")
    remaining = max(0.000001, deadline - accounted_through)
    session, progress, reused = None, [], set()
    status, code, pause = "not_started", 0, None
    try:
        session = ledger.SearchSession(args.output, plan,
            session_budget_seconds=remaining,
            campaign_budget_seconds=plan["initial_controls"]["campaign_budget_seconds"],
            max_storage_bytes=10 * 1024**3,
            clock=HandoffClock(accounted_through, clock))
        session._start_session()
        paths = session.pending_record_paths
        mirror.push(args.output, paths=paths, message="Resume unchanged eleven-origin selection after registered recovery")
        session.mark_records_synced(paths)
        if args.repair_only:
            status = "recovery_completed"
        else:
            wrapped = DeadlineSession(session, deadline, clock)
            def worker(job, bound):
                if on_new_fit is not None:
                    on_new_fit()
                return optimized.isolated_trial(plan, legacy_plan, job,
                    min(bound, deadline), args, runner)
            optimized.run_selection(plan, legacy_plan, legacy, wrapped, panel, runner,
                                    worker, mirror, progress, reused)
            status = "completed_eleven_origin_selection_not_final_benchmark"
    except ledger.SessionPaused as error:
        status, pause = "paused_" + error.reason, error.reason
    except ledger.UnresolvedAttempt:
        status, code = "reconciliation_required", 3
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    except Exception:
        status, code = "blocked_runtime_failure", 2
        raise
    finally:
        if session is not None:
            session.close()
            report = runner.report(plan, session, status, progress, REPOSITORY, BRANCH,
                                   pause_reason=pause)
            report.update(reused_jobs=len(reused),
                total_available_unique_jobs=plan["legacy_cache"]["completed_jobs"] + len(session.completed_job_ids),
                combined_charged_seconds=plan["legacy_cache"]["spent_seconds"] + session.charged_wall_seconds)
            runner.write_json(Path(args.output) / "status.json", report)
            mirror.push(args.output, paths=(*session.pending_record_paths, "status.json"),
                        message="Save optimized selection after one registered recovery")
    return status, code


def execute(args, *, clock=time.monotonic):
    started = clock()
    deadline = started + 28800.0
    mirrors = []
    legacy = transport = None
    report = {"version": "e2-optimized-recovery-controller-v1", "status": "not_started",
              "training_started": False,
              "final_market_values_read": False, "final_R": None}
    try:
        assert_no_other_executor()
        registered, optimized, runner, public, ledger, Mirror, core, Transport = load_registered(args)
        mirror = Mirror(REPOSITORY, BRANCH, BRANCH)
        mirrors.append(mirror)
        restored = mirror.restore(args.output)
        config = json.loads((Path(args.output) / "campaign.json").read_bytes())
        plan = config["plan"]
        if digest(plan) != PLAN or restored.get("plan_sha256") != PLAN:
            raise ValueError("sealed_optimized_plan_changed")
        public._validate_plan(plan)
        transport = Transport()
        saved_raw = transport.read("authorization.json")
        saved = None if saved_raw is None else json.loads(saved_raw)
        if saved is not None:
            completed_state = core.RecoveryReceiptStore(transport, saved).inspect()
            if completed_state.get("completed") is not None:
                report.update(status="already_recovered_use_normal_controller",
                    recovered_job_id=saved["target"]["job_id"],
                    combined_total_hours_authorized=saved["limits"]["combined_total_hours"])
                return report, 0
        view = core.read_snapshot(args.output, plan, session_class=ledger.SearchSession,
                                  authorization=saved)
        old_charged = view.charged_wall_seconds
        combined = plan["legacy_cache"]["spent_seconds"] + old_charged
        report.update(status="inspection_only", completed_jobs=len(view.completed_job_ids),
            inspection_pending_job_ids=list(view.unresolved_job_ids),
            combined_conservative_charged_seconds=combined,
            current_combined_total_hours=(plan["legacy_cache"]["spent_seconds"] + view._loader.campaign_budget_seconds) / 3600,
            remaining_seconds=view._loader.campaign_budget_seconds - old_charged,
            required_reservation_seconds=view.target_start["reservation_seconds"],
            recovery_authorization_exists=saved is not None,
            registered_total_hours=None if saved is None else saved["limits"]["combined_total_hours"],
            source_campaign_commit=restored.get("remote_commit"))
        if args.inspect:
            return report, 0
        if not args.execute:
            raise ValueError("explicit_execute_required")
        if saved is None and restored.get("remote_commit") != SEALED_HEAD:
            raise ValueError("sealed_campaign_head_changed")
        if runner.cuda_hardware() != plan["hardware_binding"]:
            raise ValueError("registered_cuda_environment_changed")
        if saved is None:
            if args.total_hours is None:
                raise ValueError("explicit_new_combined_total_required")
            authorization = core.build_authorization(view, total_hours=args.total_hours,
                implementation_sha256=registered["implementation_sha256"])
        else:
            authorization = saved
            if args.total_hours is not None and not math.isclose(args.total_hours, saved["limits"]["combined_total_hours"], abs_tol=1e-9):
                raise ValueError("registered_recovery_total_cannot_be_changed")
        receipts = core.RecoveryReceiptStore(transport, authorization)
        state = receipts.inspect()
        if state.get("started") is not None and state.get("observed") is None:
            raise ValueError("recovery_started_without_observed_outcome")
        old = Mirror(REPOSITORY, optimized.OLD_BRANCH, optimized.OLD_BRANCH)
        mirrors.append(old)
        old.restore(args.legacy_cache)
        if old._head != plan["legacy_cache"]["commit"]:
            raise ValueError("legacy_snapshot_changed")
        legacy_plan = json.loads((Path(args.legacy_cache) / "campaign.json").read_bytes())["plan"]
        if digest(legacy_plan) != optimized.LEGACY_PLAN:
            raise ValueError("legacy_plan_changed")
        legacy = ledger.SearchSession(args.legacy_cache, legacy_plan,
            session_budget_seconds=28800, campaign_budget_seconds=86400,
            max_storage_bytes=10 * 1024**3)
        reconciler = core.ManualReconciler(view, authorization, mirror, receipts,
            authorization_sha256=digest(authorization), recovery_source_commit=args.recovery_commit,
            physical_session_started_at=started, physical_deadline=deadline, clock=clock,
            executor_absence_confirmed=True)
        panel, _ = runner.prepare_panel(args.data_root)
        replay = ReplayFacade(view, authorization, reconciler)
        def worker(job, bound):
            assert_no_other_executor()
            expected_head = mirror._head
            mirror.inspect_remote()
            if mirror._head != expected_head:
                raise ValueError("campaign_head_changed_immediately_before_recovery_fit")
            report["training_started"] = True
            print(json.dumps({"event": "registered_recovery_fit_started", "model": job["model"],
                              "origin": job["origin"], "seed": job["seed"], "stage": job["stage"]}), flush=True)
            return optimized.isolated_trial(plan, legacy_plan, job, min(bound, deadline), args, runner)
        try:
            optimized.run_selection(plan, legacy_plan, legacy, replay, panel, runner,
                                    worker, mirror, [], set())
        except RecoveryStopped as stopped:
            result = stopped.result
        else:
            raise ValueError("sealed_recovery_target_not_reached")
        report.update(status="recovery_completed", recovered_job_id=result["job_id"],
            original_physical_attempt_still_unknown=True,
            combined_total_hours_authorized=authorization["limits"]["combined_total_hours"],
            completed_jobs=result["completed_jobs"])
        def new_fit_started():
            report["training_started"] = True
        status, code = continue_selection(args, optimized, runner, ledger, plan,
            legacy_plan, legacy, mirror, panel, deadline,
            result["campaign_accounted_through_monotonic"], clock=clock,
            on_new_fit=new_fit_started)
        report["status"] = status
        final = json.loads((Path(args.output) / "status.json").read_bytes())
        if final.get("plan_sha256") != PLAN:
            raise ValueError("continuation_report_plan_changed")
        report.update(completed_jobs=final["completed_jobs"],
            combined_conservative_charged_seconds=final["combined_charged_seconds"],
            current_combined_total_hours=(plan["legacy_cache"]["spent_seconds"] +
                final["budget"]["campaign_budget_seconds"]) / 3600,
            remaining_seconds=final["budget"]["remaining_campaign_seconds"])
        return report, code
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="blocked_recovery", error_code=str(error)[:250],
                      error_type=type(error).__name__)
        return report, 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        if legacy is not None:
            legacy.close()
        if transport is not None:
            transport.close()
        for mirror in mirrors:
            mirror.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--optimization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root-itransformer", type=Path, required=True)
    parser.add_argument("--recovery-commit", required=True)
    parser.add_argument("--total-hours", type=float)
    parser.add_argument("--repair-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() in args.report.resolve().parents or args.report.exists():
        raise ValueError("fresh_external_report_path_required")
    lock_path = args.output.parent / ".optimized-recovery.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report, code = execute(args)
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
