"""Bounded orchestration: cached replay, foreign jobs, physical deadlines and accounting."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_e2_optimized_recovery_v1 as r
import e2_search_session as ledger


class ReplayTests(unittest.TestCase):
    def fixture(self):
        payload = {'model': 'fixture', 'seed': 1}
        plan = r.digest({'fixture_plan': True})
        job = r.digest({'plan_sha256': plan, 'job': payload})
        auth = {'target': {'job_id': job, 'job_payload_sha256': r.digest(payload), 'reservation_seconds': 10}}
        class View:
            directory = Path('/unused')
            plan_sha256 = plan
            def get_result(self, identity, payload=None):
                return {'cached': True} if identity == 'cached' else None
        reconciler = SimpleNamespace(recover=lambda *a, **kw: {'job_id': job})
        return r.ReplayFacade(View(), auth, reconciler), job, payload

    def test_cached_results_never_call_worker(self):
        facade, _, _ = self.fixture()
        result = facade.run_job('cached', {}, lambda _: self.fail('unexpected fit'), reservation_seconds=10)
        self.assertEqual(result, {'cached': True})
        with self.assertRaisesRegex(ValueError, 'unexpected_uncached'):
            facade.run_job('foreign', {}, lambda _: self.fail('foreign fit'), reservation_seconds=10)

    def test_only_bound_target_is_recovered_once(self):
        facade, job, payload = self.fixture()
        with self.assertRaises(r.RecoveryStopped) as stopped:
            facade.run_job(job, payload, lambda _: None, reservation_seconds=10)
        self.assertEqual(stopped.exception.result['job_id'], job)
        with self.assertRaisesRegex(ValueError, 'payload_changed'):
            facade.run_job(job, payload, lambda _: self.fail('second fit'), reservation_seconds=10)

    def test_wrong_payload_cannot_reach_reconciler(self):
        facade, job, _ = self.fixture()
        facade.reconciler.recover = lambda *a, **k: self.fail('unbound retry')
        with self.assertRaises(ValueError):
            facade.run_job(job, {'changed': True}, lambda _: None, reservation_seconds=10)

    def test_deadline_refuses_job_before_session_admission(self):
        class Session:
            def get_result(self, *args): return None
            def _pause(self, reason, **kw): raise ledger.SessionPaused(reason, kw)
            def run_job(self, *args, **kw): raise AssertionError('job was admitted')
        session = r.DeadlineSession(Session(), 100, lambda: 95)
        with self.assertRaises(ledger.SessionPaused):
            session.run_job('x', {}, lambda _: None, reservation_seconds=10)

    def test_clock_handoff_includes_post_commit_time(self):
        now = [110.0]
        clock = r.HandoffClock(100.0, lambda: now[0])
        self.assertEqual(clock(), 100.0)
        self.assertEqual(clock(), 110.0)
        now[0] = 115.0
        self.assertEqual(clock(), 115.0)

    def test_own_pid_allowed_but_other_executor_blocks(self):
        with patch.object(r.subprocess, 'run', return_value=SimpleNamespace(stdout=f'{r.os.getpid()} python run_e2_optimized_recovery_v1.py\n')):
            r.assert_no_other_executor()
        with patch.object(r.subprocess, 'run', return_value=SimpleNamespace(stdout='999999 python run_e2_optimized_selection_v2.py\n')):
            with self.assertRaisesRegex(ValueError, 'another_executor'):
                r.assert_no_other_executor()

    def test_continuation_reuses_remaining_cap_and_charges_ack_gap(self):
        with tempfile.TemporaryDirectory() as d:
            now = [110.0]
            root = Path(d) / 'campaign'
            plan = {'initial_controls': {'campaign_budget_seconds': 90000},
                    'legacy_cache': {'completed_jobs': 0, 'spent_seconds': 0}}
            captured = {}
            class Optimized:
                @staticmethod
                def run_selection(plan, old_plan, legacy, session, panel, runner, worker, mirror, progress, reused):
                    captured['session_budget'] = session.session_budget_seconds
                    self.assertEqual(session.session_budget_seconds, 100.0)
                    session.run_job('fixture', {'fixture': True}, lambda deadline: worker({}, deadline), reservation_seconds=50)
                @staticmethod
                def isolated_trial(*args):
                    now[0] = 120.0
                    return {'status': 'completed'}
            class Mirror:
                def push(self, *args, **kwargs): return {'remote_commit': 'a' * 40}
            class Runner:
                @staticmethod
                def report(plan, session, status, *args, **kwargs):
                    return {'plan_sha256': session.plan_sha256, 'completed_jobs': len(session.completed_job_ids),
                            'budget': session.summary()}
                @staticmethod
                def write_json(path, value): path.write_text(json.dumps(value))
            args = SimpleNamespace(output=root, repair_only=False)
            marks = []
            status, code = r.continue_selection(args, Optimized, Runner, ledger, plan, {}, None,
                Mirror(), None, 200.0, 100.0, clock=lambda: now[0], on_new_fit=lambda: marks.append(True))
            self.assertEqual(code, 0)
            self.assertEqual(status, 'completed_eleven_origin_selection_not_final_benchmark')
            self.assertEqual(marks, [True])
            recorded = json.loads((root / 'status.json').read_text())
            self.assertAlmostEqual(recorded['budget']['charged_wall_seconds'], 20.0)


if __name__ == '__main__':
    unittest.main()
