"""Scientific-estimand and dependence guards; no model fitting or ETTh1 access."""
import sys
from pathlib import Path
import unittest
import json
import copy
import tempfile
import os
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
from analyze_e2_paper_guide import (seed_mean_interval,replication_summary,
    circular_block_indices,sparse_to_calendar,bootstrap_distribution,
    centered_upper_p,holm,region_gap,analyse_limitation,validate_analysis_design,
    adjust_region_family,plots,clean_json)
from paper_guide_pilot_diagnostics import input_variability


class PaperStatisticsTests(unittest.TestCase):
    @staticmethod
    def synthetic_diagnostics(high_mask):
        n=len(high_mask);queries=np.arange(0,n,8,dtype=np.int64)
        result={}
        for name,magnitude in [('TimeMixer',.02),('iTransformer',1e-8),('PatchTST',2e-8)]:
            result[name]={"query_indices":queries,"calendar_length":n,
                          "high_mask":np.asarray(high_mask,dtype=bool),
                          "batch_delta":np.full((5,len(queries)),magnitude),
                          "no_marks_delta":np.full((5,len(queries)),1e-8),
                          "query_MSE_B1":np.ones((5,len(queries))),
                          "query_MSE_bank":np.full((5,len(queries)),1.1),
                          "val_squared_error_by_origin":np.tile(np.linspace(.2,.7,n),(5,1))}
        return result

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

    def test_empty_or_all_high_strata_are_null_inconclusive_without_aborting_L1(self):
        for high in [np.zeros(384,dtype=bool),np.ones(384,dtype=bool)]:
            with self.subTest(all_high=bool(high.all())):
                result=analyse_limitation(self.synthetic_diagnostics(high),blocks=(192,),repetitions=99)
                block=result['by_block']['192']
                self.assertEqual(len(block['L1_batch_sensitivity']),3)
                self.assertGreater(block['L1_batch_sensitivity'][0]['effect'],0)
                for row in block['L2_high_input_variability']:
                    self.assertEqual(row['status'],'not_estimable')
                    self.assertEqual(row['conclusion'],'inconclusive')
                    for key in ['effect','ci95','p_one_sided_unadjusted','p_one_sided_Holm_family3','positive_gap_supported']:
                        self.assertIsNone(row[key])
                    self.assertEqual(row['registered_hypothesis_family_size'],3)
                    self.assertFalse(row['threshold_and_block_length_changed'])
                serialized=json.dumps(clean_json(result),allow_nan=False)
                self.assertIn('"effect": null',serialized)

    def test_insufficient_estimable_bootstrap_strata_remain_inconclusive(self):
        mask=np.zeros(384,dtype=bool);mask[0]=True
        result=analyse_limitation(self.synthetic_diagnostics(mask),blocks=(192,),repetitions=99)
        rows=result['by_block']['192']['L2_high_input_variability']
        for row in rows:
            self.assertEqual(row['reason'],'insufficient_estimable_bootstrap_draws')
            self.assertTrue(row['bootstrap_attempted'])
            self.assertLess(row['finite_bootstrap_draws'],.95*99)
            self.assertIsNone(row['effect']);self.assertIsNone(row['ci95'])
        self.assertTrue(result['by_block']['192']['L1_paired_control_contrasts'])

    def test_nonestimable_hypothesis_does_not_reduce_Holm_family(self):
        records=[{'p_one_sided_unadjusted':.01,'ci95':[.1,.2]},
                 {'p_one_sided_unadjusted':None,'ci95':None},
                 {'p_one_sided_unadjusted':.04,'ci95':[.1,.2]}]
        adjust_region_family(records)
        self.assertAlmostEqual(records[0]['p_one_sided_Holm_family3'],.03)
        self.assertAlmostEqual(records[2]['p_one_sided_Holm_family3'],.08)
        self.assertIsNone(records[1]['p_one_sided_Holm_family3'])
        self.assertIsNone(records[1]['p_one_sided_unadjusted'])
        self.assertEqual([r['registered_hypothesis_family_size'] for r in records],[3,3,3])

    def test_invalid_error_arrays_still_fail_instead_of_becoming_missing_strata(self):
        data=self.synthetic_diagnostics(np.zeros(384,dtype=bool))
        data['TimeMixer']['val_squared_error_by_origin'][0,0]=-1
        with self.assertRaisesRegex(ValueError,'negative'):analyse_limitation(data,blocks=(192,),repetitions=99)

    def test_secondary_MSE_change_retains_improvement_and_deterioration_without_pvalues(self):
        data=self.synthetic_diagnostics(np.zeros(384,dtype=bool))
        data['iTransformer']['query_MSE_bank'][:]=.8
        result=analyse_limitation(data,blocks=(192,),repetitions=99)
        rows=result['by_block']['192']['L1_query_MSE_change_bank_minus_B1_descriptive']
        by_name={r['model']:r for r in rows}
        self.assertAlmostEqual(by_name['TimeMixer']['effect'],.1)
        self.assertAlmostEqual(by_name['iTransformer']['effect'],-.2)
        for row in rows:
            self.assertEqual(len(row['ci95']),2)
            self.assertFalse(any(key.startswith('p_') for key in row))
            self.assertIn('no p-value',row['role'])

    def test_nonestimable_regime_plot_keeps_replica_L1_and_marks_NE_without_zero_effects(self):
        result=analyse_limitation(self.synthetic_diagnostics(np.zeros(384,dtype=bool)),blocks=(192,),repetitions=99)
        replica=replication_summary([.374,.375,.376,.375,.375],[.399,.4,.401,.4,.4])
        os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-e2-statistics-tests')
        os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-e2-statistics-tests')
        import matplotlib.axes
        original=matplotlib.axes.Axes.errorbar
        calls=[]
        def record(axis,*args,**kwargs):
            calls.append(args);return original(axis,*args,**kwargs)
        with tempfile.TemporaryDirectory(prefix='synthetic-e2-statistics-',dir='/tmp') as directory:
            with patch.object(matplotlib.axes.Axes,'errorbar',new=record):plots(Path(directory),replica,result)
            self.assertEqual(len(calls),4)  # replica + three L1; no invented L2 point
            self.assertTrue((Path(directory)/'replication_discrepancy.png').is_file())
            self.assertTrue((Path(directory)/'pilot_batch_invariance.png').is_file())
            self.assertTrue((Path(directory)/'pilot_regime_error.png').is_file())


if __name__=="__main__":unittest.main()
