"""Explicit metadata amendments against real local Git; never market or GPU."""
from contextlib import contextmanager
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if not (ROOT / "scripts/e2_search_session.py").is_file():
    ROOT = HERE.parent / "e2opt"
SCRIPT = ROOT / "scripts/extend_e2_optimized_budget.py"
if not SCRIPT.is_file():
    SCRIPT = HERE / "extend_e2_optimized_budget.py"
spec = importlib.util.spec_from_file_location("budget_extension_subject", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sys.path.insert(0, str(ROOT / "scripts"))
import run_e2_optimized_selection_v2 as optimized
import e2_github_campaign as public
from e2_github_git_transport import GitCampaignMirror
from e2_search_session import SearchSession, SessionPaused

optimized.install_public_schema(public)


def git(*args):
    return subprocess.run(["git", *map(str, args)], check=True,
                          capture_output=True, text=True).stdout.strip()


def plan():
    rows = [{"width": w, "batch": b, "num_kernels": 6, "parameters": 10,
             "original_seconds": 2., "cropped_seconds": 1., "speedup": 2.,
             "forward_and_gradient_parity": True}
            for w in (16, 32, 64, 128, 256) for b in (16, 32, 64)]
    profile = {"version": "timesnet-support-crop-v1", "device": "cuda:0",
        "torch": "2.8.0+cu126", "device_name": "test_device", "repeats": 5,
        "synthetic_only": True, "market_data_read": False,
        "optimizer_included": False, "end_to_end_training_speedup_measured": False,
        "rows": rows, "fast_cases": [[r["width"], r["batch"]] for r in rows],
        "cuda_gate_passed": True}
    return {"version": optimized.VERSION, "source_commit": optimized.ENGINE,
        "protocol_sha256": "a" * 64, "registration_sha256": "b" * 64,
        "source_sha256": {"scripts/frozen.py": "c" * 64},
        "origins": list(optimized.ORIGINS), "models": ["TimesNet"], "k": 40,
        "hardware_binding": {"device_name": "test_device"},
        "initial_controls": {"session_budget_seconds": 28800.,
            "campaign_budget_seconds": 172700., "storage_bytes": 10 * 1024**3},
        "remote_storage": {"repository": optimized.REPOSITORY,
            "branch": optimized.BRANCH, "prefix": optimized.BRANCH},
        "optimization": {"profile": profile, "profile_sha256": optimized.digest(profile),
            "fast_cases": profile["fast_cases"]},
        "legacy_cache": {"commit": "d" * 40, "plan_sha256": optimized.LEGACY_PLAN,
            "completed_jobs": 1802, "spent_seconds": 100., "limit_seconds": 172800.}}


class BudgetExtensionTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, open_session=False):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            remote = base / "remote.git"
            git("init", "--bare", remote)
            value = plan()
            registration = {"implementation_sha256": value["source_sha256"]}
            opt = types.SimpleNamespace(REPOSITORY=optimized.REPOSITORY,
                BRANCH=optimized.BRANCH, VERSION=optimized.VERSION,
                read_registration=lambda: (deepcopy(registration), "b" * 64))
            def factory(repository, branch, prefix):
                return GitCampaignMirror(repository, branch, prefix,
                    token="test-only-token", _test_remote=remote,
                    _allow_local_for_tests=True)
            def state(path, supplied_plan=value, **kwargs):
                if not kwargs:
                    kwargs = dict(session_budget_seconds=28800.,
                        campaign_budget_seconds=172700., max_storage_bytes=10 * 1024**3)
                return SearchSession(path, supplied_plan, **kwargs)
            clock = [100.]
            with factory(opt.REPOSITORY, opt.BRANCH, opt.BRANCH) as target:
                path = base / "initial"
                target.restore(path)
                with SearchSession(path, value, session_budget_seconds=28800.,
                        campaign_budget_seconds=172700., max_storage_bytes=10 * 1024**3,
                        clock=lambda: clock[0]) as session:
                    def callback(deadline):
                        clock[0] += 5
                        return {"status": "completed", "run_id": "e" * 64,
                            "resource_use": {"optimizer_steps": 1},
                            "predictions_original": [.1],
                            "prediction_dates": ["2024-01-01T00:00:00+00:00"],
                            "phase_epochs": {"supervised": 1}, "reason": None, "fatal": False}
                    session.run_job("cached", {"seed": 1}, callback, reservation_seconds=10)
                    try:
                        session._pause("campaign_budget")
                    except SessionPaused:
                        pass
                    if open_session:
                        target.push(path, paths=session.pending_record_paths)
                if not open_session:
                    status = {"version": "e2-cuda-search-runner-v1",
                        "plan_sha256": session.plan_sha256, "status": "paused_campaign_budget",
                        "completed_jobs": 1, "failed_jobs": 0, "budget": session.summary(),
                        "pause_reason": "campaign_budget", "combined_charged_seconds": 105.}
                    (path / "status.json").write_text(json.dumps(status))
                    target.push(path, paths=(*session.pending_record_paths, "status.json"))
            original_config = (base / "initial/campaign.json").read_bytes()
            args = types.SimpleNamespace(total_hours=56., confirmed_by_user=True)
            yield base, remote, opt, factory, state, value, original_config, args

    def test_confirmation_and_finite_hours_are_required_before_reading_or_writing(self):
        def forbidden(*args, **kwargs):
            self.fail("unconfirmed extension consulted remote or registration")
        opt = types.SimpleNamespace(read_registration=forbidden)
        with self.assertRaisesRegex(ValueError, "confirmation_required"):
            helper.apply_extension(types.SimpleNamespace(total_hours=56, confirmed_by_user=False),
                                   opt, forbidden, forbidden)
        for value in (True, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                helper.checked_hours(value, True)

    def test_additive_extension_roundtrips_preserves_plan_charge_results_and_is_idempotent(self):
        with self.fixture() as (base, remote, opt, factory, state, value, original, args):
            result = helper.apply_extension(args, opt, factory, state)
            self.assertEqual(result["status"], "budget_amendment_confirmed")
            self.assertEqual(result["added_hours"], 8.)
            self.assertFalse(result["training_started"])
            head = result["remote_commit"]
            with factory(opt.REPOSITORY, opt.BRANCH, opt.BRANCH) as target:
                target.restore(base / "restored")
                self.assertEqual((base / "restored/campaign.json").read_bytes(), original)
                with state(base / "restored") as session:
                    self.assertEqual(session.plan, value)
                    self.assertEqual(session.campaign_budget_seconds, 56 * 3600 - 100)
                    self.assertEqual(session.charged_wall_seconds, 5.)
                    self.assertEqual(len(session.completed_job_ids), 1)
                    self.assertEqual(sum(e["kind"] == "campaign_budget_extended" for e in session._events), 1)
                    outcome = session.run_job("cached", {"seed": 1},
                        lambda deadline: self.fail("completed work repeated"), reservation_seconds=10)
                    self.assertEqual(outcome["status"], "completed")
            again = helper.apply_extension(args, opt, factory, state)
            self.assertEqual(again["status"], "already_registered")
            self.assertEqual(again["remote_commit"], head)
            self.assertEqual(git("--git-dir=" + str(remote), "rev-parse", "refs/heads/" + opt.BRANCH), head)

    def test_open_session_allocation_requires_reconciliation_without_remote_amendment(self):
        with self.fixture(open_session=True) as (base, remote, opt, factory, state, value, original, args):
            head = git("--git-dir=" + str(remote), "rev-parse", "refs/heads/" + opt.BRANCH)
            with self.assertRaisesRegex(ValueError, "requires_reconciliation"):
                helper.apply_extension(args, opt, factory, state)
            self.assertEqual(git("--git-dir=" + str(remote), "rev-parse", "refs/heads/" + opt.BRANCH), head)

    def test_lower_total_limit_is_refused_without_remote_amendment(self):
        with self.fixture() as (base, remote, opt, factory, state, value, original, args):
            head = git("--git-dir=" + str(remote), "rev-parse", "refs/heads/" + opt.BRANCH)
            args.total_hours = 47.
            with self.assertRaisesRegex(ValueError, "reduction_refused"):
                helper.apply_extension(args, opt, factory, state)
            self.assertEqual(git("--git-dir=" + str(remote), "rev-parse", "refs/heads/" + opt.BRANCH), head)

    def test_push_failure_never_starts_training_and_retry_after_lost_ack_is_idempotent(self):
        with self.fixture() as (base, remote, opt, factory, state, value, original, args):
            class LostAcknowledgement(GitCampaignMirror):
                def push(self, *a, **kw):
                    super().push(*a, **kw)
                    raise public.CampaignSyncError("simulated_lost_ack")
            def failing_factory(repository, branch, prefix):
                return LostAcknowledgement(repository, branch, prefix, token="test-only-token",
                    _test_remote=remote, _allow_local_for_tests=True)
            with self.assertRaisesRegex(public.CampaignSyncError, "simulated_lost_ack"):
                helper.apply_extension(args, opt, failing_factory, state)
            result = helper.apply_extension(args, opt, factory, state)
            self.assertEqual(result["status"], "already_registered")
            self.assertFalse(result["training_started"])
            with factory(opt.REPOSITORY, opt.BRANCH, opt.BRANCH) as target:
                target.restore(base / "after-failure")
                with state(base / "after-failure") as session:
                    self.assertEqual(len(session.completed_job_ids), 1)
                    self.assertEqual(session.charged_wall_seconds, 5.)
                    self.assertEqual(sum(e["kind"] == "campaign_budget_extended" for e in session._events), 1)

    def test_helper_source_has_no_training_or_session_start_call(self):
        forbidden = {"_start_session", "run_job", "run_selection", "isolated_trial", "prepare_panel", "execute"}
        for node in ast.walk(ast.parse(SCRIPT.read_text())):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main()
