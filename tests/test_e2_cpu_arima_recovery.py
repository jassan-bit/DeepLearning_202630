"""Recovery command boundaries; no inference, optimizer or budget calls."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/e2_cpu_arima_worker.py'
SPEC = importlib.util.spec_from_file_location('recovery_worker_gate', PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class RecoveryCommandGate(unittest.TestCase):
    def setUp(self):
        self.plan = {'original_driver': {'path': '/registered/driver.py'},
                     'original_plan': {'path': '/registered/plan.json'},
                     'remaining_job_ids': ['ARIMA', 'CPU-freeze-raw-score', 'CPU-primary20000'],
                     'new_output': '/new/v4', 'frozen_source_root': '/source'}
        self.argv = ['/registered/driver.py', '--worker', '--job', 'ARIMA',
                     '--output', '/new/v4', '--plan', '/registered/plan.json',
                     '--source-root', '/source']

    def test_only_three_remaining_jobs_are_permitted(self):
        for name in self.plan['remaining_job_ids']:
            self.argv[3] = name
            worker.check_command(self.argv, self.plan)
        self.argv[3] = 'FCN-seed271801'
        with self.assertRaisesRegex(ValueError, 'No neural refitting'):
            worker.check_command(self.argv, self.plan)

    def test_original_output_cannot_be_modified(self):
        self.argv[5] = '/old/v3'
        with self.assertRaisesRegex(ValueError, 'output identity'):
            worker.check_command(self.argv, self.plan)

    def test_other_plan_cannot_be_substituted(self):
        self.argv[7] = '/unregistered/plan.json'
        with self.assertRaisesRegex(ValueError, 'plan path'):
            worker.check_command(self.argv, self.plan)

    def test_parent_run_cannot_use_worker_adapter(self):
        self.argv.remove('--worker')
        with self.assertRaisesRegex(ValueError, 'worker invocation only'):
            worker.check_command(self.argv, self.plan)


if __name__ == '__main__':
    unittest.main()
