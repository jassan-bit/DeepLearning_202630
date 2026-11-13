"""Resource accounting must preserve scope and reject broken evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import collect_e2_efficiency_evidence as audit


def fit_fixture(sae=False):
    model = 'LSTM-autoencoder con predictor' if sae else 'MLP16'
    request = {'name': model, 'seed': 271801, 'trial': {'trial_id': 'registered'},
               'resources': {'device_name': 'fixture GPU', 'dtype': 'float32'},
               'sources': {'source_commit': audit.SCIENTIFIC_COMMIT}}
    phase = {'elapsed_seconds': 2., 'known_cost_seconds': 2., 'optimizer_steps': 10,
             'epochs_completed': 5, 'fixed_epochs': 5, 'cuda_synchronized_timing': True,
             'runtime': {'cuda_peak_scope': 'since_callers_last_reset'}}
    phases = {'supervised': deepcopy(phase)}
    if sae:
        phases.update(ae1=deepcopy(phase), ae2=deepcopy(phase))
    fit = {'status': 'completed', 'model_id': model, 'seed': 271801,
           'synthetic': False, 'training_performed': True, 'test_only_cpu_execution': False,
           'request_sha256': audit.sha(audit.canonical(request)),
           'selected_checkpoint': {'sha256': 'f'*64},
           'architecture': {'actual_parameter_count': 87},
           'phase_records': phases, 'optimizer_steps': 10*len(phases)}
    return fit, request


class ResourceEvidenceTests(unittest.TestCase):
    def test_sae_phases_are_additive_without_inventing_missing_memory_or_latency(self):
        fit, request = fit_fixture(sae=True)
        rows = audit.extract_fit(fit, request, 'final', 'attempt', 'final/path')
        self.assertEqual({r['phase'] for r in rows}, {'ae1','ae2','supervised'})
        self.assertEqual(sum(r['elapsed_seconds'] for r in rows), 6)
        self.assertEqual(sum(r['optimizer_steps'] for r in rows), 30)
        self.assertTrue(all(r['max_cuda_memory_allocated_bytes'] is None for r in rows))
        self.assertTrue(all(r['inference_seconds'] is None for r in rows))

    def test_tampered_fit_request_or_frozen_epoch_role_rejected(self):
        fit, request = fit_fixture()
        request['seed'] = 271802
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit.extract_fit(fit, request, 'final', 'attempt', 'final/path')
        fit, request = fit_fixture()
        fit['phase_records']['supervised']['fixed_epochs'] = None
        with self.assertRaisesRegex(ValueError, 'frozen epoch'):
            audit.extract_fit(fit, request, 'final', 'attempt', 'final/path')

    def test_nonfinite_measurement_and_step_mismatch_rejected(self):
        fit, request = fit_fixture()
        fit['phase_records']['supervised']['elapsed_seconds'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            audit.extract_fit(fit, request, 'final', 'attempt', 'final/path')
        fit, request = fit_fixture()
        fit['optimizer_steps'] = 11
        with self.assertRaisesRegex(ValueError, 'reconcile'):
            audit.extract_fit(fit, request, 'final', 'attempt', 'final/path')

    def test_pareto_uses_both_coordinates_excludes_missing_and_keeps_ties(self):
        rows = [{'model':'cheap','cost':1,'RMSE':3}, {'model':'good','cost':3,'RMSE':1},
                {'model':'dominated','cost':4,'RMSE':2}, {'model':'tie','cost':3,'RMSE':1},
                {'model':'deterministic','cost':None,'RMSE':0}]
        self.assertEqual(audit.pareto(rows,'cost'),
                         {'cheap':True,'good':True,'dominated':False,'tie':True})
        self.assertIsNone(audit.strict_sum([1,None]))
        self.assertIsNone(audit.strict_sum([]))

    def test_closed_ledger_reservations_do_not_become_actual_cost(self):
        previous, rows = '0'*64, []
        for body in [{'kind':'reserved','attempt_id':'a','reservation_seconds':100},
                     {'kind':'charged','attempt_id':'a','elapsed_seconds':3},
                     {'kind':'overhead','elapsed_seconds':2}]:
            row = {**body, 'sequence':len(rows), 'previous_sha256':previous}
            row['sha256'] = audit.sha(audit.canonical(row)); previous = row['sha256']; rows.append(row)
        raw = b'\n'.join(audit.canonical(row) for row in rows)+b'\n'
        budget = {'ledger_source_bytes':len(raw), 'ledger_source_sha256':audit.sha(raw),
                  'terminal_event_sha256':previous, 'current_ledger_charged_and_overhead_seconds':5}
        result = audit.audit_ledger(raw,budget)
        self.assertEqual(result['current_worker_charged_seconds'],3)
        self.assertEqual(result['current_controller_overhead_seconds'],2)
        self.assertEqual(result['reservation_sum_seconds_not_actual_cost'],100)
        self.assertIsNone(result['monetary_cost'])
        damaged = raw.replace(b'"elapsed_seconds":3', b'"elapsed_seconds":9')
        with self.assertRaisesRegex(ValueError,'SHA256'):
            audit.audit_ledger(damaged,budget)
        # Updating the outer file hash cannot conceal a broken internal chain.
        with self.assertRaisesRegex(ValueError,'hash chain'):
            audit.audit_ledger(damaged,{**budget,'ledger_source_sha256':audit.sha(damaged)})

    def test_classical_fit_exists_while_resource_measurements_remain_unknown(self):
        quality = [{'model':'ARIMA','RMSE':'0.01','RMSE_seed_SD':'','R':'1','n_common_dates':'608'}]
        row = audit.summarize([],quality)[0]
        self.assertEqual(row['final_fit_count'],1)
        for key in ('final_phase_seconds_sum','actual_parameter_count','final_optimizer_steps_sum',
                    'pareto_RMSE_vs_measured_final_phase_time'):
            self.assertIsNone(row[key])

    def test_all_six_frozen_seeds_required_before_model_aggregation(self):
        fit, request = fit_fixture()
        phases = audit.extract_fit(fit,request,'final','attempt','final/path')
        quality = [{'model':'MLP16','RMSE':'0.01','RMSE_seed_SD':'0.001','R':'6','n_common_dates':'608'}]
        with self.assertRaisesRegex(ValueError,'final seed'):
            audit.summarize(phases,quality)


if __name__ == '__main__':
    unittest.main()
