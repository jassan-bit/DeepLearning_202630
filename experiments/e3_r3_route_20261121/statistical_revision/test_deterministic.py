"""Deterministic software guards only: no RNG or Monte Carlo calibration."""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
from scipy import stats
import crossed_rmse_v2_frozen as frozen
from revised_primary import Settings, absolute_tail, from_indices, run
from dgp_revision import bases_for, feature_phis
from run_validation import CP, decision, final_state, summarize


def fixture():
    s = np.arange(12)[:, None, None]
    t = np.arange(32)[None, :, None]
    m = np.arange(4)[None, None, :]
    return (1+.015*s+.03*m+.12*np.sin((t+2*m+s)/7))**2


class ArithmeticStream:
    """Fixed arithmetic arrays implementing only the consumed RNG interface."""
    def __init__(self):
        self.call = 0

    def integers(self, high, size):
        self.call += 1
        indices = np.arange(np.prod(size) if isinstance(size, tuple) else size)
        out = (indices*indices+3*indices+7*self.call) % high
        return out.reshape(size).astype(np.int64)

    def random(self, size):
        self.call += 1
        return ((np.arange(size)*7+self.call) % 29)/29


class Guards(unittest.TestCase):
    settings = Settings(B=63, mean_block=6., hac_lag=4, batch=8)
    names = ['candidate', 'a', 'b', 'c']

    def call(self, q=None):
        return run(fixture() if q is None else q, self.names, 0, ArithmeticStream(), self.settings)

    def test_target_mean_individual_rmse(self):
        np.testing.assert_allclose(self.call()['risks'], np.sqrt(fixture().mean(axis=1)).mean(axis=0), rtol=1e-14)

    def test_seed_only_reduces_to_paired_mean_studentizer(self):
        a = np.column_stack([1+np.arange(12)/30, 1.1+np.arange(12)**2/300,
                             1.2+np.arange(12)/40, 1.3+np.arange(12)/50])
        q = np.repeat(a[:, None, :]**2, 32, axis=1)
        result = self.call(q)
        d = a[:, 1]-a[:, 0]
        self.assertAlmostEqual(result['pairwise'][0]['theta_hat'], d.mean(), places=14)
        self.assertAlmostEqual(result['pairwise'][0]['SE'], d.std(ddof=1)/np.sqrt(12), places=13)
        self.assertLess(result['pairwise'][0]['time_variance'], 1e-28)

    def test_frozen_control_exact_same_arithmetic_draws(self):
        old = frozen.run(fixture(), self.names, 0, ArithmeticStream(), self.settings)
        new = self.call()
        self.assertEqual(new['MCS_bootstrap_t_frozen_control'], old['MCS_bootstrap_t'])
        for a, b in zip(new['pairwise'], old['pairwise']):
            for key in ['theta_hat', 'SE', 'P2_equal_two_sided_p', 'P2_CI95',
                        'P2_SE', 'seed_variance', 'time_variance']:
                self.assertEqual(a[key], b[key])
            self.assertEqual(a['P1_basic_frozen_control_p'], b['P1_equal_two_sided_p'])
            self.assertEqual(a['P1_basic_frozen_control_CI95'], b['P1_CI95'])
            self.assertEqual(a['P1_basic_frozen_control_Holm38'], b['P1_Holm'])
            self.assertEqual(a['P2_frozen_control_Holm38'], b['P2_Holm'])

    def test_strict_discrete_tail_boundary(self):
        pivots = np.arange(1., 1000.)[:, None]
        for observed, expected_p, excluded in [(950.5, .05, False), (951., .05, False), (951.5, .049, True)]:
            p, critical, rank = absolute_tail(np.array([observed]), pivots, .05)
            self.assertEqual(rank, 951)
            self.assertEqual(critical[0], 951.)
            self.assertEqual(p[0], expected_p)
            self.assertEqual(bool(abs(observed) > critical[0]), excluded)
            self.assertEqual(bool(p[0] < .05), excluded)

    def test_ties_are_conservative(self):
        p, critical, rank = absolute_tail(np.array([2.]), np.full((999, 1), 2.), .05)
        self.assertEqual(p[0], 1.)
        self.assertEqual(critical[0], 2.)

    def test_empirical_p_floor_and_unresolved_interval(self):
        p, critical, rank = absolute_tail(np.array([5.]), np.zeros((63, 1)), .001)
        self.assertEqual(p[0], 1/64)
        self.assertTrue(np.isinf(critical[0]))
        self.assertEqual(rank, 64)

    def test_studentization_uses_each_bootstrap_scale(self):
        p, critical, _ = absolute_tail(np.array([2.]), np.array([[1.], [3.], [-4.]]), .4)
        self.assertEqual(p[0], .75)
        self.assertEqual(critical[0], 4.)

    def test_deterministic_replay(self):
        self.assertEqual(self.call(), self.call())

    def test_scale_invariance(self):
        a, b = self.call(), self.call(fixture()*1e-20)
        for x, y in zip(a['pairwise'], b['pairwise']):
            self.assertEqual(x['P1_equal_two_sided_p'], y['P1_equal_two_sided_p'])
            self.assertAlmostEqual(x['P2_equal_two_sided_p'], y['P2_equal_two_sided_p'], places=10)
        np.testing.assert_allclose(np.asarray(b['risks'])/1e-10, a['risks'], rtol=1e-12)

    def test_equal_zero_variance_reserves_family(self):
        result = self.call(np.ones((12, 32, 4)))
        self.assertEqual(len(result['pairwise']), 3)
        for row in result['pairwise']:
            self.assertEqual(row['P1_equal_two_sided_p'], 1.)
            self.assertEqual(row['P1_Holm'], 1.)
            self.assertIsNone(row['P1_CI95'])
        self.assertEqual(result['MCS_bootstrap_t_frozen_control']['retained'], self.names)

    def test_unequal_zero_variance_is_not_certainty(self):
        q = np.broadcast_to(np.array([1., 4., 9., 16.]), (12, 32, 4)).copy()
        result = self.call(q)
        self.assertTrue(all(x['P1_CI95'] is None and x['P2_CI95'] is None for x in result['pairwise']))
        self.assertTrue(all(x['P1_equal_two_sided_p'] == 1. for x in result['pairwise']))

    def test_bootstrap_zero_scale_nonzero_root_reserved(self):
        seeds = np.zeros((63, 12), dtype=int)
        dates = np.tile(np.arange(32), (63, 1))
        q = np.repeat((1+np.arange(12)[:, None]*np.array([[.01, .03, .04, .05]]))[:, None, :]**2, 32, axis=1)
        result = from_indices(q, self.names, 0, seeds, dates, self.settings)
        for row in result['pairwise']:
            self.assertFalse(row['P1_supported'])
            self.assertEqual(row['P1_equal_two_sided_p'], 1.)
            self.assertIsNone(row['P1_CI95'])
            self.assertEqual(row['P1_bootstrap_unsupported_draws'], 63)

    def test_no_market_or_third_method_accreditation(self):
        result = self.call()
        self.assertFalse(result['R3_accredited'])
        self.assertFalse(result['third_primary_method_accredited'])
        self.assertEqual(result['new_market_fits'], 0)
        self.assertEqual(result['new_market_predictions'], 0)

    def test_invalid_inputs(self):
        for q in [fixture()[:9], fixture()[:, :31], np.zeros((12, 32, 4))]:
            with self.assertRaises(ValueError):
                self.call(q)
        for bad in [np.nan, np.inf, -1.]:
            q = fixture()
            q[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                self.call(q)

    def test_invalid_indices(self):
        with self.assertRaises(ValueError):
            from_indices(fixture(), self.names, 0, np.full((63, 12), 12),
                         np.tile(np.arange(32), (63, 1)), self.settings)

    def test_declared_partial_nulls_are_real_equality_nulls(self):
        plan = json.loads((Path(__file__).parent/'VALIDATION_PLAN.json').read_text())
        case = next(c for c in plan['cases'] if c['id'] == 'partial_equality')
        theta = bases_for(case)[1:]-bases_for(case)[0]
        self.assertEqual(int((theta == 0).sum()), 9)
        self.assertEqual(int((theta > .03).sum()), 10)

    def test_heterogeneous_phi_is_between_models(self):
        plan = json.loads((Path(__file__).parent/'VALIDATION_PLAN.json').read_text())
        case = next(c for c in plan['cases'] if c['id'] == 'null_heterogeneous_model_temporal_dependence')
        phis = feature_phis(case)
        self.assertEqual(phis.shape, (1+20+12+12*20,))
        self.assertEqual(phis[1], .85)
        self.assertEqual(phis[2], 0.)
        np.testing.assert_array_equal(bases_for(case), np.ones(20))

    def test_holm_never_drops_unsupported_slot(self):
        np.testing.assert_allclose(frozen.holm([.001, 1., .02, .4]), [.004, 1., .06, .8])

    def test_CP_and_decision_do_not_certify_market(self):
        self.assertEqual(CP(0, 256)['CI95'][0], 0.)
        self.assertGreater(CP(14, 128)['CI95'][0], .05)
        self.assertFalse(decision([], False)['R3_accredited'])
        self.assertEqual(decision([], False)['status'], 'incomplete_no_family_approval')

    def test_pointwise_CI_does_not_claim_simultaneous(self):
        result = self.call()
        case = {'id': 'guard_only', 'R': 12, 'T': 32, 'M': 4, 'bases': [1., 1., 1.23, 1.23],
                'replicates': 1, 'delta': .03}
        summary = summarize(case, [{'result': result}])
        self.assertFalse(summary['simultaneous_CI95_claimed'])
        self.assertEqual(len(summary['true_null_rivals']), 1)
        self.assertIn('known_true_null_two_method_intersection_event', summary)
        self.assertIn('any_true_best_excluded', summary['MCS_bootstrap_t_frozen_control'])

    def test_unsupported_nonnull_still_blocks_family_screen(self):
        result = copy.deepcopy(self.call())
        result['pairwise'][1]['P1_CI95'] = None
        result['pairwise'][1]['P1_supported'] = False
        case = {'id': 'guard_only', 'R': 12, 'T': 32, 'M': 4, 'bases': [1., 1., 1.23, 1.23],
                'replicates': 1, 'delta': .03}
        summary = summarize(case, [{'result': result}])
        self.assertEqual(summary['known_true_null_component_FWER_Holm38']['screen'],
                         'inconclusive_unsupported_replicates')

    def test_frozen_source_change_prevents_completion(self):
        self.assertEqual(final_state(1664, [], True), (True, 'completed'))
        self.assertEqual(final_state(1664, [], False), (False, 'error'))
        self.assertEqual(final_state(128, [], True), (False, 'resource_stop'))
        self.assertEqual(final_state(128, [{'type': 'guard_error'}], True), (False, 'error'))

    def test_estimand_counterexample_preserved(self):
        self.assertLess(np.sqrt([.01, 8.41]).mean(), np.sqrt([3., 3.]).mean())
        self.assertGreater(np.mean([.01, 8.41]), np.mean([3., 3.]))


if __name__ == '__main__':
    unittest.main()
