"""Scientific-estimand and dependence guards; no model fitting or ETTh1 access."""
import sys
from pathlib import Path
import unittest
import json
import copy
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
from analyze_e2_paper_guide import (seed_mean_interval,replication_summary,
    circular_block_indices,sparse_to_calendar,bootstrap_distribution,
    centered_upper_p,holm,region_gap,analyse_limitation,validate_analysis_design)
from paper_guide_pilot_diagnostics import input_variability


class PaperStatisticsTests(unittest.TestCase):
    def test_seed_compatibility_uses_published_MSE_units_not_BTC_delta(self):
        r=replication_summary([.374,.375,.376,.375,.375],[.399,.400,.401,.400,.400])
        self.assertTrue(r["registered_success_rule"]["numerical_compatibility"])
        self.assertAlmostEqual(r["registered_success_rule"]["absolute_margin"],.01875)
        self.assertFalse(r["Bitcoin_margin_used"])

    def test_interval_containing_reference_is_not_replication_equivalence(self):
        r=replication_summary([.1,.2,.375,.55,.65],[.2,.3,.4,.5,.6])
        self.assertLess(r["MSE"]["ci95"][0],.375)
        self.assertGreater(r["MSE"]["ci95"][1],.375)
        self.assertFalse(r["registered_success_rule"]["numerical_compatibility"])

    def test_single_seed_cannot_supply_conditional_training_uncertainty(self):
        with self.assertRaises(ValueError):seed_mean_interval([.375])

    def test_published_point_is_not_assigned_an_original_variance(self):
        r=replication_summary([.375]*5,[.4]*5)
        self.assertEqual(r["reference"]["publication_dispersion_of_selected_row"],"NR")
        self.assertTrue(r["reference"]["reference_treated_as_fixed_rounded_value"])
        self.assertEqual(r["MSE"]["difference_ci95"],[0.,0.])
        self.assertTrue(r["registered_success_rule"]["does_not_by_itself_establish_identical_historical_row_conditions"])

    def test_sparse_queries_preserve_full_hourly_calendar(self):
        x=sparse_to_calendar([[1.,2.,3.],[4.,5.,6.]],[0,200,999],1000)
        self.assertEqual(x.shape,(2,1000));self.assertTrue(np.isnan(x[:,1:200]).all())
        np.testing.assert_array_equal(x[:,[0,200,999]],[[1,2,3],[4,5,6]])
        with self.assertRaises(ValueError):sparse_to_calendar([1,2],[4,4],1000)

    def test_circular_blocks_retain_adjacent_hourly_origins(self):
        idx=circular_block_indices(103,12,np.random.default_rng(42))
        self.assertEqual(len(idx),103)
        for start in range(0,96,12):
            np.testing.assert_array_equal(np.diff(idx[start:start+12])%103,np.ones(11))

    def test_pairing_exact_difference_has_no_artificial_variance(self):
        a=np.linspace(.01,.09,200)
        diff=np.stack([a-a,a-a])
        b=bootstrap_distribution(diff,lambda x,i:float(x.mean()),32,199,76)
        np.testing.assert_array_equal(b,np.zeros(199))
        self.assertEqual(centered_upper_p(0,b),1.)

    def test_frontier_test_uses_registered_nonzero_tolerance(self):
        b=np.linspace(.09,.11,199)
        p0=centered_upper_p(.1,b,.1)
        p1=centered_upper_p(.1,b,1e-5)
        self.assertGreater(p0,.4);self.assertLess(p1,.01)

    def test_holm_preserves_input_order_and_adjusts_all_three(self):
        np.testing.assert_allclose(holm([.04,.001,.02]),[.04,.003,.04])
        with self.assertRaises(ValueError):holm([.2,1.1])

    def test_regime_effect_averages_seed_RMSE_after_nonlinear_transform(self):
        e=np.array([[1.,1.,4.,4.],[9.,9.,16.,16.]])
        r=region_gap(e,[False,False,True,True])
        self.assertAlmostEqual(r["effect"],1.)
        self.assertAlmostEqual(r["high_RMSE_mean"],3.)
        self.assertNotAlmostEqual(r["high_RMSE_mean"],float(np.sqrt(e[:,2:].mean())))

    def test_empty_region_and_negative_squared_error_are_not_zero(self):
        with self.assertRaises(ValueError):region_gap([[1.,2.]],[True,True])
        with self.assertRaises(ValueError):region_gap([[1.,-2.]],[True,False])

    def test_three_actual_pilot_models_are_required(self):
        with self.assertRaises(ValueError):analyse_limitation({"TimeMixer":{}},repetitions=99)

    def test_rolling_context_variability_uses_only_eligible_histories(self):
        rng=np.random.default_rng(31);values=rng.normal(size=(30,7))
        class D:
            seq_len=6
            def __len__(self):return 15
        d=D();d.values=values
        actual=input_variability(d)
        expected=[np.sqrt(np.mean(np.diff(values[i:i+6],axis=0)**2)) for i in range(15)]
        np.testing.assert_allclose(actual,expected)
        self.assertEqual(len(actual),15)

    def test_sparse_bootstrap_draws_come_from_real_dense_hours(self):
        dense=sparse_to_calendar([1.,2.,3.,4.],[0,50,100,150],200)
        seen=[]
        def stat(x,index):
            seen.append(len(index));return float(np.nanmean(x))
        b=bootstrap_distribution(dense,stat,40,99,99)
        self.assertTrue(all(n==200 for n in seen));self.assertEqual(len(b),99)

    def test_registered_design_rejects_changed_metric_units_and_sparse_blocks(self):
        root=Path(__file__).resolve().parents[1]
        design=json.loads((root/"experiments/e2_paper_guide_20261106/design.json").read_text())
        validate_analysis_design(design)
        changed=copy.deepcopy(design);changed["replication_criterion"]["absolute_tolerance"]=.00025772842664981106
        with self.assertRaises(ValueError):validate_analysis_design(changed)
        changed=copy.deepcopy(design);changed["analysis"]["block_lengths"]=[36,18,72]
        with self.assertRaises(ValueError):validate_analysis_design(changed)


if __name__=="__main__":unittest.main()
