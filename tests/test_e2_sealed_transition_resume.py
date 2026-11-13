"""Resume sealed historical witnesses after later cache additions; no training."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import e2_sealed_transition_resume as S
import test_e2_a100_cache_selection as fixtures
Q = fixtures.M
E = Q.original


class SealedResumeTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.MigrationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.prepared()
        self.args, self.budget = self.f.args, self.f.budget
        self.receipt_path = self.args.hardware_transition_receipt
        receipt = json.loads(self.receipt_path.read_bytes())
        receipt['registration_commit'] = S.Q_COMMIT
        receipt.pop('sha256')
        receipt['sha256'] = E.digest(receipt)
        self.receipt_path.write_bytes(E.canonical(receipt) + b'\n')
        self.args.hardware_transition_registration_commit = S.Q_COMMIT
        self.old_completed = self.budget.completed_receipt
        self.final_proofs = {}
        for name in ['FCN-1D', 'ResNet-1D', 'Inception-1D']:
            for seed in range(271801, 271807):
                directory = self.f.output / 'final/final-fit' / E.digest(name)[:16] / str(seed)
                directory.mkdir(parents=True)
                (directory / 'weights.pt').write_bytes((name + str(seed)).encode())
                selection = {'winner_configuration_id': 'C01', 'frozen': True}
                unsigned = {'version': S.FINAL_VERSION, 'kind': 'neural-final',
                    'source_identity': {'source_commit': S.ENGINE_COMMIT},
                    'runtime_identity': deepcopy(self.f.actual['runtime']),
                    'selection': selection, 'model': name, 'seed': seed}
                spec_sha = E.digest(unsigned)
                E.write_json(directory / 'worker_spec.json', {**unsigned, 'worker_spec_sha256': spec_sha})
                result = {'status': 'completed', 'worker_spec_sha256': spec_sha,
                    'model': name, 'seed': seed, 'selection_sha256': E.digest(selection),
                    'artifacts_sha256': {'weights.pt': E.file_sha(directory / 'weights.pt')}}
                E.write_json(directory / 'result.json', result)
                logical = E.digest({'version': S.FINAL_VERSION, 'spec_sha256': spec_sha})
                self.final_proofs[logical] = {'worker_spec_sha256': spec_sha,
                    'result_sha256': E.file_sha(directory / 'result.json')}
        def completed(identity, *, expected_resource_use):
            if identity in self.final_proofs:
                self.assertEqual(expected_resource_use, self.final_proofs[identity])
                return {'sha256': 'f' * 64}
            return self.old_completed(identity, expected_resource_use=expected_resource_use)
        self.budget.completed_receipt = completed
        self.completed = completed
        # Existing funding/selection metadata must remain untouched too.
        E.write_json(self.f.output / 'shared_budget/budget_continuation_80.json', {'effective_total_hours': 80})
        E.write_json(self.f.output / 'selection_complete.json', {'status': 'completed'})

    def run_resume(self, **kwargs):
        with patch.object(E, 'source_identity', return_value=self.f.source):
            return S.prepare_sealed_historical_cache(self.args, self.budget,
                registration_commit=S.Q_COMMIT, resume_registration_commit='e' * 40, **kwargs)

    def snapshot(self):
        return {p.relative_to(self.f.output).as_posix(): p.read_bytes()
            for p in self.f.output.rglob('*') if p.is_file()}

    def test_later_earlier_sorted_attestation_is_ignored_and_all_18_fits_survive(self):
        before_profile = json.loads(self.args.historical_runtime_profile.read_bytes())
        directory = self.f.output / 'jobs' / ('0' * 64)
        directory.mkdir()
        E.write_json(directory / 'cache_admission.json', self.f.admission)
        E.write_json(directory / 'result.json', {'status': 'completed', 'cache_reuse': True,
            'artifacts_sha256': {'cache_admission.json': E.file_sha(directory / 'cache_admission.json')}})
        plans = [{'hardware_binding': fixtures.runtime('Tesla T4'), 'execution': self.f.policy}] * 2
        with patch.object(E, 'source_identity', return_value=self.f.source):
            regenerated = Q._historical_profile(self.f.output, self.budget, plans)
        self.assertNotEqual(regenerated, before_profile)
        before = self.snapshot()
        with patch.object(Q, '_historical_profile', side_effect=AssertionError('Never regenerate')):
            result = self.run_resume()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(len(result['resume_admission']['verified_completed_final_fits']), 18)
        self.assertTrue(result['resume_admission']['current_observation_is_original_profile_file'])
        self.assertFalse(result['resume_admission']['historical_files_modified'])

    def test_fresh_wrapper_metadata_uses_separate_fix_receipt_without_changing_original(self):
        actual = deepcopy(self.f.actual)
        actual['fresh_diagnostic_wrapper'] = 'current-observation'
        self.f.runtime_path.write_bytes(E.canonical(actual) + b'\n')
        before = self.snapshot()
        value = self.run_resume()['resume_admission']
        self.assertFalse(value['current_observation_is_original_profile_file'])
        self.assertEqual(value['resume_registration_commit'], 'e' * 40)
        self.assertNotEqual(value['current_A100_profile_sha256'], value['original_A100_profile_sha256'])
        self.assertEqual(before, self.snapshot())
        with patch.object(E, 'source_identity', return_value=self.f.source):
            with self.assertRaisesRegex(ValueError, 'distinct registered receipt'):
                S.prepare_sealed_historical_cache(self.args, self.budget, registration_commit=S.Q_COMMIT)

    def test_corrupted_profile_and_audit_block_without_writing(self):
        for path in [self.args.historical_runtime_profile, self.args.cache_audit]:
            raw = path.read_bytes()
            path.write_bytes(raw + b' ')
            before = self.snapshot()
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                self.run_resume()
            self.assertEqual(before, self.snapshot())
            path.write_bytes(raw)

    def test_unknown_historical_or_final_cost_blocks(self):
        for scope in ['historical', 'final']:
            def completed(identity, *, expected_resource_use):
                if (identity in self.final_proofs) == (scope == 'final'):
                    return None
                return self.completed(identity, expected_resource_use=expected_resource_use)
            self.budget.completed_receipt = completed
            with self.assertRaisesRegex(ValueError, 'budget charge'):
                self.run_resume()

    def test_actual_runtime_change_or_policy_change_blocks(self):
        actual = deepcopy(self.f.actual)
        actual['runtime']['total_memory_bytes'] += 1
        self.f.runtime_path.write_bytes(E.canonical(actual) + b'\n')
        with self.assertRaisesRegex(ValueError, 'A100 identity differs'):
            self.run_resume()
        actual = deepcopy(self.f.actual)
        actual['fit_execution_policy']['changed'] = True
        self.f.runtime_path.write_bytes(E.canonical(actual) + b'\n')
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            self.run_resume()

    def test_changed_final_weights_and_wrong_q_registration_block(self):
        path = next((self.f.output / 'final/final-fit').glob('*/*/weights.pt'))
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact is missing or changed'):
            self.run_resume()
        with self.assertRaisesRegex(ValueError, 'original Q'):
            S.prepare_sealed_historical_cache(self.args, self.budget, registration_commit='d' * 40)


if __name__ == '__main__':
    unittest.main()
