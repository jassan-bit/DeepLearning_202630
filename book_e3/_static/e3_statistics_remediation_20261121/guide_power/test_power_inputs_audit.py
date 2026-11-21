"""Integrity and scope guard fixtures; no market or Monte Carlo experiment."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from power_inputs_audit import assess_contract, linearized_budget_diagnostic, safe_file


class PowerInputGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        version = 'fixture_candidate_not_a_real_model'
        scope = dict(candidate_version=version, estimand='mean_individual_RMSE', units='sigma7',
                     seed_uncertainty='population_initializations', calendar_uncertainty='future_process')
        self.config = {**scope, 'candidate_status': 'viable_for_planning',
            'family': {'rivals': ['r1', 'r2'], 'methods': ['method1', 'method2', 'method3'],
                       'contrastive_mapping_justified': True},
            'panels': [{'panel_id': 'F'+str(i), 'admitted_unexposed': True,
                        'excluded_from_confirmation': True, 'practical_margin': .1,
                        'planning_alternative': .2} for i in range(4)],
            'planning_seed_pairs': 5, 'final_R': 10, 'final_T': 128,
            'budget': {'max_wall_seconds': 10, 'max_replicates': 100},
            'persistence': {'verified_checkpoint_required': True, 'skip_verified_complete': True,
                            'overwrite_existing': False},
            'controlled_validation_protocol_sha256': '1'*64,
            'full_decision_rule_sha256': '2'*64, 'evidence_bindings': {}}
        covariance = [{**scope, 'panel_id': p['panel_id'], 'rival': rival,
                       'scope': 'prospective_planning', 'paired_seed_pairs': 5,
                       'seed_variance': .1, 'calendar_lrv': .2, 'interaction_lrv': .3,
                       'cross_rival_covariance_sha256': '3'*64}
                      for p in self.config['panels'] for rival in ['r1', 'r2']]
        self.receipts = {
            'scientific_decision': {'candidate_version': version, 'planning_allowed': True,
                                    'current_recipe_abandoned': False},
            'admission': {'admitted_panel_ids': ['F0', 'F1', 'F2', 'F3'], 'datasets_admitted': 4,
                          'all_planning_partitions_excluded_from_confirmation': True},
            'paired_covariance': {**scope, 'cells': covariance},
            'calibration': {**scope, 'approved_for_scope': True,
                            'methods': ['method1', 'method2', 'method3'],
                            'controlled_validation_protocol_sha256': '1'*64},
            'power': {**scope, 'selected_R': 10, 'selected_T': 128,
                      'full_decision_rule_sha256': '2'*64,
                      'uses_observed_confirmation_results': False,
                      'complete_three_procedure_rule': True, 'independent_MC_replicates': 1000,
                      'per_contrast': [{'panel_id': p['panel_id'], 'rival': rival,
                                        'simultaneous_MC_lower': .91}
                                       for p in self.config['panels'] for rival in ['r1', 'r2']],
                      'joint_power': .2, 'joint_MC_interval': [.1, .3]}}
        for key in self.receipts:
            self.bind(key)

    def bind(self, key):
        data = json.dumps(self.receipts[key], sort_keys=True).encode()
        (self.root / (key+'.json')).write_bytes(data)
        self.config['evidence_bindings'][key] = {'path': key+'.json',
                                               'sha256': hashlib.sha256(data).hexdigest()}

    def report(self):
        return assess_contract(self.config, self.root)

    def test_consistent_fixture_never_authorizes_or_certifies(self):
        r = self.report()
        self.assertTrue(all(r['checks'].values()), r['blockers'])
        self.assertFalse(r['confirmation_enabled'])
        self.assertFalse(r['power_certified_by_this_program'])
        self.assertFalse(r['can_train'])

    def test_joint90_is_not_required(self):
        self.assertTrue(self.report()['checks']['joint_power_and_precision_reported_without_joint90_rule'])

    def test_incompatible_model_version(self):
        self.receipts['paired_covariance']['candidate_version'] = 'other_model'
        self.bind('paired_covariance')
        self.assertFalse(self.report()['checks']['paired_covariance_same_version_and_estimand'])

    def test_conditional_MSE_does_not_fill_primary_RMSE_scope(self):
        self.receipts['calibration']['estimand'] = 'fixed_cohort_mean_MSE'
        self.bind('calibration')
        self.assertFalse(self.report()['checks']['calibration_same_version_and_estimand'])

    def test_conditional_seeds_not_population_initializations(self):
        self.receipts['calibration']['seed_uncertainty'] = 'fixed_cohort'
        self.bind('calibration')
        self.assertFalse(self.report()['checks']['calibration_same_version_and_estimand'])

    def test_exposed_development_covariance_cannot_be_promoted(self):
        self.receipts['paired_covariance']['cells'][0]['scope'] = 'exposed_development'
        self.bind('paired_covariance')
        self.assertFalse(self.report()['checks']['complete_compatible_seed_calendar_covariance'])

    def test_power_point_estimate_does_not_replace_MC_lower(self):
        self.receipts['power']['per_contrast'][0]['simultaneous_MC_lower'] = .89
        self.bind('power')
        self.assertFalse(self.report()['checks']['full_rule_power_at_least90_with_MC_precision'])

    def test_observed_power_rejected(self):
        self.receipts['power']['uses_observed_confirmation_results'] = True
        self.bind('power')
        self.assertFalse(self.report()['checks']['full_rule_power_at_least90_with_MC_precision'])

    def test_future_cell_missing_does_not_become_zero(self):
        self.receipts['paired_covariance']['cells'][0]['calendar_lrv'] = None
        self.bind('paired_covariance')
        self.assertFalse(self.report()['checks']['complete_compatible_seed_calendar_covariance'])

    def test_missing_rival_not_accepted(self):
        self.receipts['paired_covariance']['cells'].pop()
        self.bind('paired_covariance')
        self.assertFalse(self.report()['checks']['complete_compatible_seed_calendar_covariance'])

    def test_hash_corruption_is_not_valid_evidence(self):
        (self.root/'power.json').write_text('{}')
        self.assertFalse(self.report()['checks']['evidence_bindings_present_and_verified'])

    def test_minimum_planning_pairs(self):
        self.config['planning_seed_pairs'] = 3
        self.assertFalse(self.report()['checks']['planning_seed_pairs_at_least5'])

    def test_minimum_final_pairs(self):
        self.config['final_R'] = 9
        self.assertFalse(self.report()['checks']['final_minimum10_and_calendar_fixed'])

    def test_budget_and_persistence_guards(self):
        self.config['budget']['max_replicates'] = 0
        self.config['persistence']['overwrite_existing'] = True
        r = self.report()
        self.assertFalse(r['checks']['computational_budget_bound'])
        self.assertFalse(r['checks']['persistence_and_no_refit_policy'])

    def test_abandoned_recipe_and_fake_admission_rejected(self):
        self.receipts['scientific_decision']['current_recipe_abandoned'] = True
        self.bind('scientific_decision')
        self.receipts['admission']['datasets_admitted'] = 0
        self.bind('admission')
        r = self.report()
        self.assertFalse(r['checks']['scientific_receipt_permits_this_planning_version'])
        self.assertFalse(r['checks']['admission_receipt_matches_four_planning_panels'])

    def test_unsafe_paths_and_symlinks_rejected(self):
        with self.assertRaises(ValueError): safe_file(self.root, '../other')
        (self.root/'alias').symlink_to(self.root/'power.json')
        with self.assertRaises(ValueError): safe_file(self.root, 'alias')

    def diagnostic(self, **changes):
        kw = dict(seed_variance=1., calendar_lrv=.1, interaction_lrv=.1, dates=128,
                  practical_margin=.1, planning_alternative=.3, multiplicity=57)
        kw.update(changes)
        return linearized_budget_diagnostic(**kw)

    def test_calendar_information_floor_blocks_any_seed_count(self):
        r = self.diagnostic(calendar_lrv=10., dates=10)
        self.assertEqual(r['status'], 'TEMPORAL_INFORMATION_FLOOR_BLOCKS_THIS_REFERENCE')
        self.assertIsNone(r['reference_R'])

    def test_more_calendar_can_reduce_reference_R(self):
        self.assertGreater(self.diagnostic()['reference_R'], self.diagnostic(dates=512)['reference_R'])

    def test_practical_margin_is_not_the_planning_alternative(self):
        with self.assertRaises(ValueError): self.diagnostic(planning_alternative=.1)
        with self.assertRaises(ValueError): self.diagnostic(planning_alternative=.05)

    def test_zero_variance_not_an_infinite_information_certificate(self):
        with self.assertRaises(ValueError):
            self.diagnostic(seed_variance=0., calendar_lrv=0., interaction_lrv=0.)

    def test_normal_reference_is_unit_covariant_and_never_final(self):
        a = self.diagnostic()
        c = 1e-6
        b = self.diagnostic(seed_variance=c*c, calendar_lrv=.1*c*c,
                            interaction_lrv=.1*c*c, practical_margin=.1*c,
                            planning_alternative=.3*c)
        self.assertEqual(a['reference_R'], b['reference_R'])
        self.assertFalse(a['complete_battery_power_computed'])
        self.assertFalse(a['final_R_selected'])
        self.assertFalse(a['confirmation_enabled'])


if __name__ == '__main__':
    unittest.main()
