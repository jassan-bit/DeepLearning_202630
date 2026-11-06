"""Explicitly synthetic tests; no market prices, S5 outcomes or model fitting."""
from __future__ import annotations
import copy
import csv
from datetime import datetime,timedelta,timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from analyze_e2_benchmark import (validate_dense_panel,regression_metrics,training_mase_scale,
    rmse_seed_mean,paired_temporal_gain,adjust_pvalues,intersection_union_p,dm_hln,
    ljung_box_diagnostics,paired_seed_diagnostics,independent_unit_global_tests,
    descriptive_ranks_and_pareto,mean_paired_seed_ranks,validate_fixture_inputs,model_registry,
    PREDICTION_FIELDS,RUN_FIELDS,COST_FIELDS)


def calendar(n):
    return [(datetime(2024,1,1,tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(n)]


def panel_rows(models=('base','candidate'),seeds=('1','2'),n=28,deterministic=()):
    rows=[]
    for m in models:
        for seed in (['deterministic'] if m in deterministic else seeds):
            for i,day in enumerate(calendar(n)):
                rows.append({'model':m,'seed':seed,'origin_utc':day,'horizon':'1','y_true':2.,
                             'y_pred':2.+(1. if m=='base' else .2),'eligible':'true',
                             'run_id':m+str(seed),'scope':'synthetic_fixture'})
    return rows


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_mean_seed_RMSE_not_pooled_or_ensemble_RMSE(self):
        rows=panel_rows(n=2)
        candidates=[r for r in rows if r['model']=='candidate']
        for r,e in zip(candidates,[-1.,-1.,0.,2.]):r['y_pred']=r['y_true']+e
        p=validate_dense_panel(rows,['base','candidate'],['1','2'],calendar(2))
        actual=rmse_seed_mean(p,'candidate')
        self.assertAlmostEqual(actual,(1+np.sqrt(2))/2)
        self.assertNotAlmostEqual(actual,np.sqrt(1.5))
        self.assertNotAlmostEqual(actual,.5)

    def test_dense_panel_rejects_missing_duplicate_and_ground_truth_mismatch(self):
        rows=panel_rows(n=3)
        for changed in [rows[:-1],rows+[rows[0]],copy.deepcopy(rows)]:
            if len(changed)==len(rows):changed[-1]['y_true']=7
            with self.assertRaises(ValueError):validate_dense_panel(changed,['base','candidate'],['1','2'],calendar(3))

    def test_missing_day_cannot_be_compressed_into_daily_calendar(self):
        days=calendar(4);days.pop(1)
        with self.assertRaisesRegex(ValueError,'compress'):validate_dense_panel(panel_rows(n=3),['base','candidate'],['1','2'],days)

    def test_ineligible_placeholders_preserve_calendar_and_pairing(self):
        rows=panel_rows(n=5)
        for r in rows:
            if r['origin_utc']==calendar(5)[2]:r['eligible']='false';r['y_pred']='';r['y_true']=''
        p=validate_dense_panel(rows,['base','candidate'],['1','2'],calendar(5))
        self.assertEqual(p['n_calendar_origins'],5);self.assertEqual(p['n_eligible_origins'],4)
        self.assertTrue(np.isnan(p['predictions'][:,:,2]).all())
        rows[-1]['eligible']='false';rows[-1]['y_pred']=''
        with self.assertRaisesRegex(ValueError,'cohort'):validate_dense_panel(rows,['base','candidate'],['1','2'],calendar(5))

    def test_negative_forecast_is_not_valid_under_D7_nonnegative_contract(self):
        rows=panel_rows(n=3);rows[0]['y_pred']=-.1
        with self.assertRaisesRegex(ValueError,'nonnegative'):validate_dense_panel(rows,['base','candidate'],['1','2'],calendar(3))

    def test_deterministic_persistence_stored_once_broadcast_not_independent_fits(self):
        rows=panel_rows(models=['base','Persistence'],n=3,deterministic=['Persistence'])
        p=validate_dense_panel(rows,['base','Persistence'],['1','2'],calendar(3),deterministic_models=['Persistence'])
        self.assertEqual(len(rows),9);self.assertEqual(p['predictions'].shape,(2,2,3))
        np.testing.assert_array_equal(p['predictions'][1,0],p['predictions'][1,1])
        self.assertEqual(p['independent_randomized_fits_for_deterministic_references'],0)
        with self.assertRaises(ValueError):validate_dense_panel(rows+rows[-3:],['base','Persistence'],['1','2'],calendar(3),deterministic_models=['Persistence'])

    def test_metric_undefined_denominators_are_not_assigned_zero(self):
        r=regression_metrics([0,0],[0,0],mase_scale=0)
        self.assertEqual(r['RMSE'],0);self.assertIsNone(r['R2']);self.assertIsNone(r['MAPE_percent'])
        self.assertIsNone(r['SMAPE_percent']);self.assertEqual(r['MAPE_coverage'],0);self.assertIsNone(r['MASE'])
        r=regression_metrics([0,2],[1,3],mase_scale=2)
        self.assertEqual(r['MAPE_defined'],1);self.assertEqual(r['MAPE_percent'],50);self.assertEqual(r['MASE'],.5)

    def test_MASE_uses_only_external_consecutive_training_pairs(self):
        r=training_mase_scale([1,2,100,3,5],calendar(5),[True,True,False,True,True])
        self.assertEqual(r['consecutive_training_pairs'],2);self.assertEqual(r['scale'],1.5)

    def test_paired_temporal_bootstrap_keeps_identical_models_identical(self):
        rows=panel_rows(n=28)
        for r in rows:r['y_pred']=3.
        p=validate_dense_panel(rows,['base','candidate'],['1','2'],calendar(28))
        r=paired_temporal_gain(p,'base','candidate',block_length=7,repetitions=199)
        self.assertEqual(r['effect'],0);self.assertEqual(r['ci95_nominal'],[0,0])
        self.assertEqual(r['p_one_sided_unadjusted'],1)
        self.assertFalse(r['seed_variability_included_as_new_market_samples'])

    def test_bootstrap_tests_delta_frontier_not_zero(self):
        p=validate_dense_panel(panel_rows(n=28),['base','candidate'],['1','2'],calendar(28))
        a=paired_temporal_gain(p,'base','candidate',block_length=7,repetitions=199,null_margin=1.)
        b=paired_temporal_gain(p,'base','candidate',block_length=7,repetitions=199,null_margin=0.)
        self.assertGreater(a['p_one_sided_unadjusted'],.9);self.assertLess(b['p_one_sided_unadjusted'],.01)
        self.assertEqual(a['declared_family_size'],30)

    def test_IUT_one_unfavorable_or_missing_component_prevents_joint_success(self):
        self.assertEqual(intersection_union_p(.001,.8)['p_IUT'],.8)
        self.assertEqual(intersection_union_p(.03,.02)['p_IUT'],.03)
        self.assertIsNone(intersection_union_p(.001,None)['p_IUT'])

    def test_missing_pvalue_kept_in_declared_family(self):
        adjusted=adjust_pvalues([.01,None,.04],method='Holm',family_size=3)
        self.assertEqual(adjusted,[.03,None,.08])
        self.assertAlmostEqual(adjust_pvalues([.01],method='Holm',family_size=30)[0],.3)
        with self.assertRaises(ValueError):adjust_pvalues([.1,.2],method='Holm',family_size=1)

    def test_DM_manual_HAC_and_HLN_h1_correction(self):
        from scipy import stats
        a=np.array([2.,3.,1.,4.,2.,5.,3.,6.]);b=np.ones(8);d=a-b;c=d-d.mean();n=len(d)
        lrv=c@c/n+2*.5*(c[1:]@c[:-1]/n)
        expected=d.mean()/np.sqrt(lrv/n)*np.sqrt((n-1)/n)
        r=dm_hln(a,b,calendar(n),horizon=1,hac_bandwidth=1)
        self.assertAlmostEqual(r['HAC_long_run_variance'],lrv);self.assertAlmostEqual(r['statistic'],expected)
        self.assertAlmostEqual(r['HLN_factor'],np.sqrt((n-1)/n))
        self.assertAlmostEqual(r['p_two_sided'],2*stats.t.sf(abs(expected),n-1))
        with self.assertRaises(ValueError):dm_hln(a,b,calendar(n),horizon=1,hac_bandwidth=0)

    def test_DM_constant_differential_is_not_infinite_significance(self):
        r=dm_hln([2]*10,[1]*10,calendar(10),horizon=1,hac_bandwidth=2)
        self.assertEqual(r['status'],'not_estimable');self.assertIsNone(r['statistic']);self.assertIsNone(r['p_two_sided'])

    def test_Ljung_residuals_and_squares_df_without_compressing_gaps(self):
        e=np.array([1.,2.,-1.,3.,-2.,2.,-1.,1.]);c=e-e.mean();rho=c[1:]@c[:-1]/(c@c);q=len(e)*(len(e)+2)*rho*rho/(len(e)-1)
        rows=ljung_box_diagnostics(e,calendar(8),lags=[1,3],model_df=0)
        self.assertEqual(len(rows),4);self.assertAlmostEqual(rows[0]['Q'],q);self.assertEqual(rows[0]['df'],1)
        self.assertEqual(rows[2]['series'],'squared_residuals')
        r=ljung_box_diagnostics([0]*8,calendar(8),lags=[1],model_df=0);self.assertIsNone(r[0]['p_unadjusted'])

    def test_conditional_seed_effect_is_paired_dz_not_market_replicates(self):
        d=np.array([.1,.2,.3,.4,.5]);r=paired_seed_diagnostics(d,unit_scope='training_seeds_conditional_fixed_calendar',repetitions=199)
        self.assertAlmostEqual(r['paired_Cohen_dz'],d.mean()/d.std(ddof=1))
        self.assertTrue(r['does_not_estimate_market_independence'])
        self.assertAlmostEqual(r['Wilcoxon_min_two_sided_p_no_ties_nonzero_exact_case'],.0625)
        constant=paired_seed_diagnostics([1]*5,unit_scope='training_seeds_conditional_fixed_calendar',repetitions=99)
        self.assertIsNone(constant['paired_Cohen_dz']);self.assertIsNone(constant['paired_Cohen_dz_ci95'])
        with self.assertRaises(ValueError):paired_seed_diagnostics(d,unit_scope='overlapping_calendar_days',repetitions=99)

    def test_global_Friedman_is_blocked_for_one_market_or_seed_rows(self):
        x=np.arange(30,dtype=float).reshape(5,6)
        with self.assertRaises(ValueError):independent_unit_global_tests(x,unit_kind='training_seeds',independent_units_asserted=True)
        with self.assertRaises(ValueError):independent_unit_global_tests(x[:1],unit_kind='independent_series',independent_units_asserted=True)

    def test_ranks_keep_valid_error_when_cost_is_missing_and_Pareto_is_descriptive(self):
        r=descriptive_ranks_and_pareto([{'model':'A','RMSE':1.,'latency_ms_per_window':None},
                                     {'model':'B','RMSE':2.,'latency_ms_per_window':2.},
                                     {'model':'C','RMSE':2.,'latency_ms_per_window':1.}])
        self.assertEqual([x['rank'] for x in r],[1.,2.5,2.5]);self.assertIsNone(r[0]['Pareto_nondominated'])
        self.assertFalse(r[1]['Pareto_nondominated']);self.assertTrue(r[2]['Pareto_nondominated'])
        ties=descriptive_ranks_and_pareto([{'model':'X','RMSE':1.,'latency_ms_per_window':1.},{'model':'Y','RMSE':1.,'latency_ms_per_window':1.}])
        self.assertTrue(all(x['Pareto_nondominated'] for x in ties))

    def test_mean_paired_seed_ranks_are_not_ranks_of_mean_RMSE(self):
        rows=panel_rows(models=['A','B'],seeds=['1','2','3'],n=2)
        for row in rows:
            error=({'1':0.,'2':0.,'3':100.}[row['seed']] if row['model']=='A' else 1.)
            row['y_pred']=row['y_true']+error
        p=validate_dense_panel(rows,['A','B'],['1','2','3'],calendar(2))
        r=mean_paired_seed_ranks(p)
        self.assertGreater(rmse_seed_mean(p,'A'),rmse_seed_mean(p,'B'))
        self.assertLess(r['mean_rank_by_model'][0],r['mean_rank_by_model'][1])
        np.testing.assert_allclose(r['mean_rank_by_model'],[4/3,5/3])
        self.assertTrue(r['does_not_supply_independent_datasets_for_Friedman'])

    def test_default_cli_generates_only_pending_templates_without_site_packages(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-analysis-docs-',dir='/tmp') as tmp:
            run=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/analyze_e2_benchmark.py'),'--output-dir',tmp],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads((Path(tmp)/'report.json').read_text());self.assertFalse(report['benchmark_executed'])
            self.assertFalse(report['numerical_rankings_published']);self.assertEqual(report['pinned_E1_source_hashes_verified'],6)
            with (Path(tmp)/'table_template.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),15);self.assertTrue(all(r['RMSE_mean']=='' and r['status'].startswith('pending') for r in rows))
            for name in ['predictions_template.csv','costs_template.csv','runs_template.csv']:
                self.assertEqual(len((Path(tmp)/name).read_text().splitlines()),1)

    def fixture_args(self, directory):
        base=Path(directory);models=[m['id'] for m in model_registry()];seeds=['1','2'];days=calendar(3)
        predictions=panel_rows(models,seeds,3,deterministic=['Persistence']);runs=[];costs=[]
        for m in models:
            for s in (['deterministic'] if m=='Persistence' else seeds):
                runs.append({'run_id':m+s,'model':m,'seed':s,'phase':'development','dataset_sha256':'a'*64,'split_sha256':'b'*64,'config_sha256':'c'*64,'checkpoint_sha256':'' if m=='Persistence' else 'd'*64,'scope':'synthetic_fixture'})
                costs.append({**{k:'' for k in COST_FIELDS},'model':m,'seed':s,'measurement_block':'0','latency_ms_per_window':'1.','device':'fixtureCPU','precision':'float32','threads':'1','batch_size':'1','input_shape':'7x5','timing_protocol_sha256':'e'*64,'scope':'synthetic_fixture'})
        for name,rows,fields in [('panel',predictions,PREDICTION_FIELDS),('runs',runs,RUN_FIELDS),('costs',costs,COST_FIELDS)]:
            with (base/(name+'.csv')).open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
        (base/'calendar.json').write_text(json.dumps({'timezone':'UTC','frequency':'D','origins':days,'seeds':seeds,'scope':'synthetic_fixture'}))
        return SimpleNamespace(panel=str(base/'panel.csv'),runs=str(base/'runs.csv'),costs=str(base/'costs.csv'),calendar=str(base/'calendar.json'),scope='synthetic_fixture',output_dir=str(base/'private_validation'))

    def test_fixture_cli_validates_hash_pairing_but_publishes_no_scores(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-benchmark-inputs-',dir='/tmp') as tmp:
            args=self.fixture_args(tmp);r=validate_fixture_inputs(args)
            self.assertTrue(r['panel_valid']);self.assertFalse(r['metric_values_published']);self.assertFalse(r['rankings_published'])
            args.scope='known_test_exploratory'
            with self.assertRaisesRegex(ValueError,'blocked'):validate_fixture_inputs(args)

    def test_fixture_outputs_rejected_in_E1_E2_E3_or_public_assembly(self):
        for output in ['book','book_e2','book_e3','_build/site']:
            args=SimpleNamespace(scope='synthetic_fixture',output_dir=str(ROOT/output))
            with self.assertRaisesRegex(ValueError,'outside the repository'):validate_fixture_inputs(args)

    def test_fixture_cost_and_run_envelope_rejects_extra_duplicate_or_unequal_sources(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-benchmark-envelope-',dir='/tmp') as tmp:
            args=self.fixture_args(tmp)
            with Path(args.costs).open() as f:rows=list(csv.DictReader(f))
            with Path(args.costs).open('a') as f:w=csv.DictWriter(f,fieldnames=COST_FIELDS);w.writerow(rows[0])
            with self.assertRaisesRegex(ValueError,'duplicated'):validate_fixture_inputs(args)
            args=self.fixture_args(tmp)
            with Path(args.runs).open() as f:rows=list(csv.DictReader(f))
            rows[-1]['split_sha256']='f'*64
            with Path(args.runs).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=RUN_FIELDS);w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'share dataset'):validate_fixture_inputs(args)


if __name__=='__main__':unittest.main()
