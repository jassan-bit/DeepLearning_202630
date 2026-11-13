"""No market access or training: preserve consent, stage order and child routes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load('notebooks/run_e2_arima_resume_colab.py', 'test_arima_continuation_launcher')
driver = load('scripts/run_e2_arima_resume_delivery.py', 'test_arima_continuation_driver')
COMMIT = 'a' * 40


class LauncherTests(unittest.TestCase):
    def test_dry_run_performs_no_fetch_consent_or_parent_work(self):
        with patch.object(launcher, '_prior_launcher') as prior, patch.object(launcher, '_materialize') as materialize:
            result = launcher.run_e2_arima_resume_colab(COMMIT)
        prior.assert_not_called()
        materialize.assert_not_called()
        self.assertFalse(result['training_started'])
        self.assertFalse(result['new_budget_authorization'])
        self.assertEqual(result['requested_stages'], ['classical', 'freeze', 'score'])

    def test_actual_caf_f80_q_h_g_route_preserves_original_authorities_and_consent(self):
        caf = load('notebooks/run_e2_final80_resume_colab.py', 'real_caf_arima_route')
        f80 = load('notebooks/run_e2_final_continuation_colab.py', 'real_f80_arima_route')
        q = load('notebooks/run_e2_a100_recovery_colab.py', 'real_q_arima_route')
        h = load('notebooks/run_e2_reduced_github72_colab.py', 'real_h_arima_route')
        g = load('notebooks/run_e2_reduced_github_colab.py', 'real_g_arima_route')
        approval = {'confirmed_text': '80', 'sha256': 'b' * 64,
            'observed_at_utc': '2026-11-13T09:54:17.605263+00:00'}
        calls = []
        g._stream = lambda command, *args: calls.append(command) or 0
        def run_g(commit, *, namespace, execute):
            g._stream(['python', Path('/Q/scripts/run_e2_reduced_github_delivery.py'),
                '--engine-root', '/E', '--extension-root', '/Q', '--extension-commit', commit],
                'env', 'cwd', 'log')
            return {'engine_commit': launcher.ENGINE_COMMIT, 'temporary_workspace': '/nonexistent'}
        g.run_reduced_github_colab = run_g
        h._base_launcher = lambda: vars(g)
        q._funding_launcher = lambda: vars(h)
        q._prepare_runtime = lambda namespace: None
        f80._prior_launcher = lambda: vars(q)
        f80._materialize = lambda commit, workspace: Path('/F80')
        caf._prior_launcher = lambda: vars(f80)
        caf._materialize = lambda commit, workspace: Path('/CAF')
        caf.existing_approval = lambda: approval
        with patch.object(launcher, '_prior_launcher', return_value=vars(caf)), \
                patch.object(launcher, '_materialize', return_value=Path('/ARIMA')):
            result = launcher.run_e2_arima_resume_colab(COMMIT, execute=True)
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertEqual(command[1], Path('/ARIMA/scripts/run_e2_arima_resume_delivery.py'))
        self.assertEqual(command[command.index('--arima-commit') + 1], COMMIT)
        self.assertEqual(command[command.index('--resume-commit') + 1], launcher.PRIOR_RESUME_COMMIT)
        self.assertEqual(command[command.index('--continuation-commit') + 1], launcher.FUNDING_CONTINUATION_COMMIT)
        self.assertEqual(command[command.index('--extension-commit') + 1], driver.PRIOR_CONTROLLER_COMMIT)
        self.assertEqual(json.loads(command[command.index('--approval-receipt') + 1]), approval)
        self.assertTrue(result['consent_reused'])
        self.assertFalse(result['new_budget_authorization'])
        self.assertFalse(result['neural_final_fits_requested'])


class RoutingTests(unittest.TestCase):
    def route(self, kind, model):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        engine, resource, repair = (root / p for p in ['E', 'P', 'ARIMA'])
        spec = root / 'worker_spec.json'
        spec.write_text(json.dumps({'kind': kind, 'model': model}))
        calls = []
        def supervisor(command, *args, **kwargs):
            calls.append((command, args, kwargs))
            return 'unchanged-supervision'
        route = driver.repair_supervisor(supervisor, engine, resource, repair, COMMIT)
        command = ['python', '-u', engine / 'scripts/e2_reduced_final.py', '--worker-spec', spec]
        return route, command, calls, engine, resource, repair

    def test_arima_and_aggregate_score_use_registered_compat_worker(self):
        for kind, model in [('classical', 'ARIMA'), ('final-score', 'all16')]:
            route, command, calls, engine, resource, repair = self.route(kind, model)
            self.assertEqual(route(command, 'budget', 'logical-id', cwd=engine), 'unchanged-supervision')
            routed, args, kwargs = calls[0]
            self.assertEqual(routed[2], repair / 'scripts/e2_arima_compat_worker.py')
            self.assertEqual(routed[routed.index('--arima-commit') + 1], COMMIT)
            self.assertEqual(routed[routed.index('--extension-commit') + 1], driver.RESOURCE_COMMIT)
            self.assertEqual(args, ('budget', 'logical-id'))
            self.assertEqual(kwargs, {'cwd': engine})

    def test_completed_non_arima_comparators_keep_the_original_p_route(self):
        route, command, calls, engine, resource, repair = self.route('classical', 'Persistence')
        route(command, 'budget')
        self.assertEqual(calls[0][0][2], resource / 'scripts/e2_resource_worker.py')
        self.assertNotIn('--arima-root', calls[0][0])

    def test_neural_and_unregistered_child_requests_are_rejected_before_supervision(self):
        route, command, calls, engine, resource, repair = self.route('neural-final', 'GRU32')
        with self.assertRaisesRegex(ValueError, 'Only comparator'):
            route(command)
        self.assertEqual(calls, [])
        command[2] = engine / 'scripts/run_e2_reduced_selection.py'
        with self.assertRaisesRegex(ValueError, 'Unexpected scientific'):
            route(command)


class CohortGuardTests(unittest.TestCase):
    def completed_fixture(self, output):
        from e2_reduced_protocol import NEURAL
        from e2_reduced_final import FINAL_SEEDS, PILOT_SEEDS, slug
        for family, models, seeds in [('final-fit', NEURAL, FINAL_SEEDS),
                ('pilot', ('iTransformer', 'GRU32'), PILOT_SEEDS)]:
            for model in models:
                for seed in seeds:
                    path = output / 'final' / family / slug(model) / str(seed) / 'result.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({'status': 'completed', 'model': model, 'seed': seed}))
        (output / 'final/pilot_power_MDE.json').write_text(json.dumps({'status': 'development_pilot_completed_power_reported'}))

    def test_only_three_remaining_stages_run_and_previous_results_are_not_rewritten(self):
        f80 = load('scripts/run_e2_final_continuation_delivery.py', 'original_f80_arima_stage_order')
        f80.FINAL_STAGES = driver.FINAL_STAGES
        from e2_reduced_protocol import NEURAL
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.completed_fixture(output)
            selected = {name: {'model_id': name, 'all_eighteen_slots_accounted_for': True,
                'validation_count': 249, 'winner_configuration_id': 'C01'} for name in NEURAL}
            (output / 'selection_complete.json').write_text(json.dumps({'status': 'completed',
                'completed_models': list(NEURAL), 'required_models': list(NEURAL), 'selections': selected}))
            before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            stages = []
            factory = driver.guarded_final_adapter(f80.final_stage_adapter)
            adapter = factory(lambda args, budget, state: stages.append(args.stage))
            adapter(SimpleNamespace(stage='all', output=output, models=None), 'budget', 'state')
            self.assertEqual(stages, ['classical', 'freeze', 'score'])
            self.assertEqual(before, {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()})

    def test_missing_final_or_pilot_blocks_before_first_remaining_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises((ValueError, FileNotFoundError)):
                driver.completed_neural_guard(output)
            self.completed_fixture(output)
            driver.completed_neural_guard(output)
            (output / 'final/pilot_power_MDE.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'completed power'):
                driver.completed_neural_guard(output)


if __name__ == '__main__':
    unittest.main()
