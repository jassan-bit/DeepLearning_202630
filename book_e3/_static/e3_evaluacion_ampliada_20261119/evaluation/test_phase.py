"""Actual optimizer/resume contracts on generated CPU data; no market inputs."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location('extended_budget_phase', Path(__file__).with_name('phase.py'))
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
torch.set_num_threads(1)

CONFIG = {'learning_rate': .01, 'weight_decay': 0., 'batch_size': 5,
          'max_epochs': 3, 'patience': 3, 'gradient_clip': 1.}


class BoundedPhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.manual_seed(302)
        self.x = torch.randn(19, 3)
        self.y = self.x[:, :1]*.6 + self.x[:, 1:2]*.2

    def tearDown(self):
        self.temp.cleanup()

    def model(self):
        return torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.2), torch.nn.Linear(5, 1))

    def run_phase(self, model, folder, **kwargs):
        return budget.phase(model, self.x, self.y, self.x, self.y, folder, CONFIG,
                            identity={'source_commit': 'a'*40, 'phase': 'synthetic'},
                            budget_seconds=60., **kwargs)

    def test_real_updates_restore_best_and_completion_retry_preserves_bytes(self):
        torch.manual_seed(22)
        model = self.model()
        result = self.run_phase(model, self.root/'one')
        self.assertEqual(result['steps'], 12)
        self.assertEqual(result['epoch'], 3)
        self.assertGreater(result['measured_completed_epoch_seconds'], 0)
        self.assertTrue(all(torch.equal(model.state_dict()[k], v) for k, v in result['best'].items()))
        marker = self.root/'one/phase_started.json'
        checkpoint = self.root/'one/checkpoint.pt'
        before = (marker.read_bytes(), checkpoint.read_bytes(), (self.root/'one/receipt.json').read_bytes())
        again = self.run_phase(self.model(), self.root/'one')
        self.assertEqual(again['steps'], result['steps'])
        self.assertEqual(before, (marker.read_bytes(), checkpoint.read_bytes(), (self.root/'one/receipt.json').read_bytes()))
        self.assertFalse(again['resource_receipt']['wall_budget_is_process_hard_timeout'])

    def test_epoch_boundary_resume_replays_same_optimizer_rng_trajectory(self):
        torch.manual_seed(11)
        full = self.run_phase(self.model(), self.root/'full')
        torch.manual_seed(11)
        save = budget._torch_save
        def interrupt(path, value):
            save(path, value)
            if value['epoch'] == 1:
                raise InterruptedError('after durable first epoch')
        with patch.object(budget, '_torch_save', side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.run_phase(self.model(), self.root/'resume')
        marker = (self.root/'resume/phase_started.json').read_bytes()
        torch.manual_seed(999)
        resumed = self.run_phase(self.model(), self.root/'resume')
        self.assertEqual(resumed['steps'], full['steps'])
        self.assertEqual(resumed['resume_segments'], 1)
        for key in full['latest']:
            self.assertTrue(torch.equal(full['latest'][key], resumed['latest'][key]), key)
        self.assertEqual(marker, (self.root/'resume/phase_started.json').read_bytes())

    def test_failure_before_first_checkpoint_keeps_immutable_attempt_marker(self):
        folder = self.root/'initial_failure'
        with patch.object(budget, '_torch_save', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):
                self.run_phase(self.model(), folder)
        marker = (folder/'phase_started.json').read_bytes()
        self.assertFalse((folder/'checkpoint.pt').exists())
        result = self.run_phase(self.model(), folder)
        self.assertGreater(result['steps'], 0)
        self.assertEqual(marker, (folder/'phase_started.json').read_bytes())

    def test_expired_untrained_checkpoint_never_becomes_a_prediction(self):
        folder = self.root/'empty'
        save = budget._torch_save
        def interrupt(path, value):
            save(path, value)
            if value['epoch'] == 0:
                raise InterruptedError('initial state only')
        with patch.object(budget, '_torch_save', side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.run_phase(self.model(), folder)
        marker = json.loads((folder/'phase_started.json').read_text())
        with patch.object(budget.time, 'time', return_value=marker['deadline_unix']+1):
            with self.assertRaises(budget.PhaseIncomplete):
                self.run_phase(self.model(), folder)
        self.assertFalse((folder/'receipt.json').exists())

    def test_expiry_after_valid_epoch_restores_it_and_does_not_reset_deadline(self):
        folder = self.root/'partial'
        save = budget._torch_save
        def interrupt(path, value):
            save(path, value)
            if value['epoch'] == 1:
                raise InterruptedError('first epoch durable')
        with patch.object(budget, '_torch_save', side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.run_phase(self.model(), folder)
        marker_raw = (folder/'phase_started.json').read_bytes()
        marker = json.loads(marker_raw)
        with patch.object(budget.time, 'time', return_value=marker['deadline_unix']+5):
            result = self.run_phase(self.model(), folder)
        self.assertEqual(result['steps'], 4)
        self.assertEqual(result['stop_reason'], 'wall_budget_expired_on_resume')
        self.assertEqual(result['resource_receipt']['deadline_overrun_seconds'], 5.)
        self.assertEqual(marker_raw, (folder/'phase_started.json').read_bytes())
        self.assertTrue(result['resource_receipt']['active_compute_lost_in_an_unclosed_crash_segment_is_unknown'])

    def test_changed_budget_or_config_cannot_reuse_old_identity(self):
        folder = self.root/'identity'
        self.run_phase(self.model(), folder)
        with self.assertRaisesRegex(ValueError, 'request differs'):
            budget.phase(self.model(), self.x, self.y, self.x, self.y, folder, CONFIG,
                         identity={'source_commit': 'a'*40, 'phase': 'synthetic'}, budget_seconds=61.)
        with self.assertRaisesRegex(ValueError, 'request differs'):
            budget.phase(self.model(), self.x, self.y, self.x, self.y, folder,
                         {**CONFIG, 'batch_size': 6},
                         identity={'source_commit': 'a'*40, 'phase': 'synthetic'}, budget_seconds=60.)

    def test_SAE_three_allocations_sum_to_one_system_budget(self):
        shares = budget.allocate_phase_budgets(90., ('ae1', 'ae2', 'supervised'))
        self.assertEqual(shares, {'ae1': 30., 'ae2': 30., 'supervised': 30.})
        self.assertEqual(sum(shares.values()), 90.)
        self.assertEqual(budget.allocate_phase_budgets(90., ('supervised',)), {'supervised': 90.})
        with self.assertRaises(ValueError):
            budget.allocate_phase_budgets(90., ('supervised', 'supervised'))

    def test_default_loss_refuses_accidental_target_broadcasting(self):
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            budget.phase(self.model(), self.x, self.y[:, 0], self.x, self.y[:, 0],
                         self.root/'broadcast', CONFIG, identity={'test': 'broadcast'}, budget_seconds=60.)

    def test_budget_stops_new_batches_and_reports_the_actual_boundary_overrun(self):
        now = [1000.]
        model = self.model()
        def loss(m, indexes):
            if indexes is None:
                return torch.mean((m(self.x)-self.y)**2)
            now[0] += 2.
            return torch.mean((m(self.x[indexes])-self.y[indexes])**2)
        with patch.object(budget.time, 'time', side_effect=lambda: now[0]):
            result = budget.phase(model, self.x, self.y, self.x, self.y,
                self.root/'short', CONFIG, identity={'fixture': 'short-boundary-budget'},
                loss_fn=loss, budget_seconds=1.)
        self.assertEqual(result['steps'], 1)
        self.assertEqual(result['curve'][0]['samples_seen'], 5)
        self.assertFalse(result['curve'][0]['full_epoch_completed'])
        self.assertEqual(result['stop_reason'], 'wall_budget')
        self.assertEqual(result['resource_receipt']['deadline_overrun_seconds'], 1.)

    def test_deadline_tampering_is_rejected_instead_of_granting_new_time(self):
        folder = self.root/'tamper'
        self.run_phase(self.model(), folder)
        marker = folder/'phase_started.json'
        value = json.loads(marker.read_text())
        value['started_unix'] += 1000.
        value['deadline_unix'] += 1000.
        marker.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Checkpoint request differs'):
            self.run_phase(self.model(), folder)

    def test_reusing_a_completed_phase_restores_rng_for_the_next_phase(self):
        torch.manual_seed(818)
        self.run_phase(self.model(), self.root/'rng')
        expected=torch.rand(12)
        torch.manual_seed(999)
        self.run_phase(self.model(), self.root/'rng')
        actual=torch.rand(12)
        self.assertTrue(torch.equal(expected,actual))


if __name__ == '__main__':
    unittest.main()
