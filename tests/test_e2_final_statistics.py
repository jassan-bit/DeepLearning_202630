"""Synthetic contracts and numerical tests; no fitting or market file access."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import copy
import json
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import e2_final_statistics as m

PREREG=json.loads((ROOT/'experiments/e2_remediation_20261106/preregistration.json').read_text())
HASH='a'*64


def prediction_fixture(kind='final',fixture=False):
    if kind=='final':
        names=m.MODELS;seeds=m.FINAL_SEEDS[:5];dates=m._calendar('2026-09-01','2026-10-30');cut='2024-12-24';stage='E2_final_primary';det=m.DETERMINISTIC
    else:
        names=m.P1_MODELS;seeds=m.P1_SEEDS;dates=m._calendar('2024-11-08','2024-12-30');cut='2024-10-31';stage='E2_P1_dimensioning';det=[]
    truth=.022+.01*np.sin(np.arange(len(dates))*.5)
    rows=[]
    for i,name in enumerate(names):
        for seed in (['deterministic'] if name in det else seeds):
            rng=np.random.default_rng(187+i+int(seed if seed!='deterministic' else 0))
            errors=rng.normal(0,.001+i*.00004,len(dates))
            for t,d in enumerate(dates):
                past=float(truth[t-1]) if t else .02
                rows.append({'model':name,'seed':seed,'outcome_day':d,'y_true':float(truth[t]),'y_pred':float(truth[t]+errors[t]),'persistence_pred':past,'high_volatility_known':past>m.THRESHOLD,'mase_scale':.001,'eligible':True,'run_id':f'{name}_{seed}','checkpoint_sha256':HASH,'scaler_sha256':HASH,'config_sha256':HASH,'checkpoint_closed_at_utc':'2026-11-06T14:00:00+00:00','scoring_started_at_utc':'2026-11-06T14:01:00+00:00','fit_outcome_max':cut,'scaler_fit_outcome_max':cut,'configuration_selection_outcome_max':cut,'source_stage':'synthetic_fixture' if fixture else stage})
    return rows


def costs_fixture(names=m.MODELS,phase='primary_final'):
    rows=[]
    for i,name in enumerate(names):
        for b in range(61):rows.append({'model':name,'block':b,'latency_ms':.4+i*.01+.002*np.sin(b+i),'phase':phase,'warmup_calls':256,'calls_per_block':256,'source_checkpoint_sha256':HASH,'parameters':0 if name in m.DETERMINISTIC else 100+i,'FLOPs_per_window':None,'peak_memory_bytes':100000,'training_seconds':10.,'hpo_seconds':20.,'hpo_seconds_kind':'actual_CPU_seconds','throughput_samples_sec':None})
    return rows


def inner_fixture(seeds=m.FINAL_SEEDS[:5]):
    rows=[];train=m._calendar('2024-01-01','2024-01-25');val=m._calendar('2024-02-02','2024-02-22')
    for i,name in enumerate(m.MODELS):
        for seed in (['deterministic'] if name in m.DETERMINISTIC else seeds):
            for split,dates in [('train',train),('validation',val)]:
                for t,d in enumerate(dates):
                    truth=.025+.003*np.sin(t);error=.001+.0001*np.sin(t+i)+(0 if split=='train' else .0004)
                    rows.append({'model':name,'seed':seed,'split':split,'outcome_day':d,'y_true':truth,'y_pred':truth+error,'eligible':True,'checkpoint_sha256':HASH,'source_stage':'E2_final_inner_diagnostics'})
    return rows


def ablation_fixture(*, production=False):
    mapping={'TimeMixer':'control','iTransformer':'ablation1','PatchTST':'ablation2','GRU/D7-c03':'ablation3'}
    rows=[];members={}
    for record in prediction_fixture('P1'):
        if record['model'] not in mapping or record['seed'] not in m.P1_SEEDS[:5]:continue
        r=dict(record);r['model']=mapping[r['model']];r['seed']=670101+r['seed']-660201
        r['source_stage']='E2_ablation_control_2024' if production else 'synthetic_fixture_ablation'
        if production:
            member=(r['model'],r['seed']);index=members.setdefault(member,len(members))
            start=datetime(2026,11,6,13,2,tzinfo=timezone.utc)+timedelta(minutes=3*index)
            r.update(ablation_schedule_sha256=HASH,ablation_schedule_closed_at_utc='2026-11-06T13:01:00+00:00',all_primary_A_weights_closed_at_utc='2026-11-06T13:00:00+00:00',ablation_fit_started_at_utc=start.isoformat(),checkpoint_closed_at_utc=(start+timedelta(minutes=1)).isoformat(),scoring_started_at_utc=(start+timedelta(minutes=2)).isoformat())
        rows.append(r)
    return rows


class FinalStatisticsTest(unittest.TestCase):
    def test_zero_variance_forecast_QLIKE_floor_and_zero_target_are_finite(self):
        result=m.volatility_scores([0.,.02],[0.,0.])
        self.assertEqual(result['QLIKE_forecast_floor_fraction'],1.)
        self.assertEqual(result['zero_target_fraction'],.5)
        self.assertTrue(np.isfinite(result['QLIKE_log_variance_plus_ratio']))
        self.assertAlmostEqual(result['MSE_realized_variance'],.5*(.02**2)**2)

    def test_constant_target_R2_bootstrap_is_not_invented(self):
        rows=prediction_fixture()
        for r in rows:
            r['y_true']=.02;r['persistence_pred']=.02;r['high_volatility_known']=False
        p=m._panel(rows,kind='final');r=m._secondary_bootstrap(p,repetitions=99,block=7,seed=5)
        self.assertTrue(all(v['R2']['ci95_nominal'] is None for v in r.values()))

    def test_settings_guard(self):
        m._settings(PREREG);j=copy.deepcopy(PREREG);j['families'][0]['alpha']=.05
        with self.assertRaises(ValueError):m._settings(j)

    def test_final_panel_20_models_seed_prefix_and_estimand(self):
        p=m._panel(prediction_fixture(),kind='final');self.assertEqual(p['predictions'].shape,(20,5,60));self.assertEqual(p['n_eligible_origins'],60)
        observed,_,_,sq=m._point(p)
        expected=np.sqrt(sq.mean(axis=-1)).mean(axis=-1)
        np.testing.assert_allclose(observed,expected)
        self.assertGreater(np.max(np.abs(observed-np.sqrt(sq.mean(axis=(1,2))))),0)

    def test_lagged_regime_rejects_target(self):
        rows=prediction_fixture()
        for r in rows:
            r['persistence_pred']=r['y_true'];r['high_volatility_known']=r['y_true']>m.THRESHOLD
        with self.assertRaisesRegex(ValueError,'previous-day'):m._panel(rows,kind='final')

    def test_cutoff_and_checkpoint_closure_guards(self):
        for field,value in [('fit_outcome_max','2025-01-01'),('checkpoint_closed_at_utc','2026-11-06T14:02:00+00:00')]:
            rows=prediction_fixture();rows[0][field]=value
            with self.assertRaises(ValueError):m._panel(rows,kind='final')

    def test_frozen_run_and_scale(self):
        rows=prediction_fixture();rows[1]['mase_scale']=.002
        with self.assertRaises(ValueError):m._panel(rows,kind='final')
        rows=prediction_fixture();rows[1]['checkpoint_sha256']='b'*64
        with self.assertRaises(ValueError):m._panel(rows,kind='final')

    def test_dense_duplicates_and_missing_seed(self):
        rows=prediction_fixture()
        with self.assertRaises(ValueError):m._panel(rows+[rows[0]],kind='final')
        with self.assertRaises(ValueError):m._panel(rows[1:],kind='final')

    def test_P1_amended_calendar(self):
        p=m._panel(prediction_fixture('P1'),kind='P1');self.assertEqual(len(p['eligible']),53)
        rows=prediction_fixture('P1');rows[0]['outcome_day']='2024-11-01'
        with self.assertRaises(ValueError):m._panel(rows,kind='P1')

    def test_timing_protocol_identity(self):
        rows=costs_fixture();self.assertEqual(m._costs(rows,m.MODELS,phase='primary_final').shape,(20,61))
        rows[0]['calls_per_block']=1
        with self.assertRaises(ValueError):m._costs(rows,m.MODELS,phase='primary_final')
        rows=costs_fixture()
        with self.assertRaises(ValueError):m._costs(rows,m.MODELS,phase='primary_final',checkpoint_hashes={name:{'b'*64} for name in m.MODELS})

    def test_bootstrap_mean_seed_recomputed(self):
        sq=np.array([[[1.,9.],[4.,4.]]]);counts=np.array([[1,0],[0,1],[1,1]])
        got=m._mean_seed_rmse(sq,counts).mean(axis=-1)
        np.testing.assert_allclose(got[:,0],[(1+2)/2,(3+2)/2,(np.sqrt(5)+2)/2])

    def test_null_practical_boundary_not_zero(self):
        boot=np.linspace(-.001,.001,10000)+.0001
        zero=m._one_sided(.0001,boot,0)['p'];practical=m._one_sided(.0001,boot,m.DELTA)['p']
        self.assertGreater(practical,zero)

    def test_sparse_resamples_not_conditioned_away(self):
        boot=np.concatenate([np.full(900,np.nan),np.linspace(.003,.004,100)])
        result=m._one_sided(.003,boot,m.DELTA);self.assertIsNone(result['p']);self.assertEqual(result['requested_draws'],1000)
        p=m._tail_p(np.r_[np.zeros(90),np.full(10,np.nan)],np.zeros(100),.001)
        self.assertTrue(np.all(p>0.1))

    def test_Holm_missing_retains_family(self):
        out=m._holm_rows([{'p':.001},{'p':None}],alpha=.02,family_size=30)
        self.assertAlmostEqual(out[0]['p_Holm'],.03);self.assertFalse(out[0]['reject']);self.assertFalse(out[1]['reject'])

    def test_wilson_is_lower_not_point_estimate(self):
        self.assertLess(m.wilson_lower(8000,10000),.8);self.assertGreater(m.wilson_lower(8200,10000),.8)

    def test_power_hypothesis_layout_and_inestimable_count(self):
        rng=np.random.default_rng(4);null=rng.normal(0,.00001,(2000,2,10,2));alt=rng.normal(0,.00001,(2000,2,10,2));cost=rng.normal(0,.000001,(2000,2,10))
        rows,decisions=m._power_cells(null,alt,cost,cost,[.4,.5],draws=2000)
        self.assertEqual(len(rows),60);self.assertEqual(decisions[0].shape,(2000,30));self.assertTrue(all(r['power']>.99 for r in rows))
        null[:]=np.nan;rows,_=m._power_cells(null,alt,cost,cost,[.4,.5],draws=2000);self.assertTrue(all(r['power']==0 for r in rows))

    def test_zero_empirical_noise_never_certifies_planning_power(self):
        p=m._tail_p(np.zeros(1000),np.zeros(1000),m.DELTA)
        self.assertTrue(np.all(p==1))
        self.assertIsNone(m._planning_MDE(np.zeros(1000),np.zeros(1000)))

    def test_MDE_cannot_discard_invalid_MC_null_draws(self):
        rng=np.random.default_rng(0);null=rng.normal(0,.001,10000);alt=rng.normal(0,.001,10000);null[:50]=np.nan
        self.assertIsNone(m._planning_MDE(null,alt))

    def test_resampling_53_to_60_preserves_counts_not_dates(self):
        counts=m._counts(53,20,7,np.random.default_rng(2),length=60);self.assertEqual(counts.shape,(20,53));np.testing.assert_array_equal(counts.sum(axis=1),np.full(20,60))

    def test_MCS_identical_losses_no_false_winner(self):
        values=np.tile(np.linspace(.001,.003,60),(3,1));result=m.model_confidence_set(values,['a','b','c'],repetitions=999)
        self.assertEqual(result['included_models'],['a','b','c']);self.assertTrue(result['separate_exploratory_family'])

    def test_inner_same_checkpoint_and_purge(self):
        rows=inner_fixture();out=m._overfit(rows,PREREG,repetitions=999,expected_seeds=m.FINAL_SEEDS[:5]);self.assertEqual(len(out['model_summaries']),20);self.assertEqual(out['formal_hypotheses'],0)
        rows[0]['checkpoint_sha256']='b'*64
        with self.assertRaises(ValueError):m._overfit(rows,PREREG,repetitions=999,expected_seeds=m.FINAL_SEEDS[:5])

    def test_inner_missing_seed_not_silently_reduced(self):
        rows=[r for r in inner_fixture() if not(r['model']=='TimeMixer' and r['seed']==670101)]
        with self.assertRaises(ValueError):m._overfit(rows,PREREG,repetitions=999,expected_seeds=m.FINAL_SEEDS[:5])

    def test_variance_partition_excludes_four_deterministic_seed_replicas(self):
        p=m._panel(prediction_fixture(),kind='final');r=m.variance_partition(p);self.assertEqual(r['model_count'],16);self.assertAlmostEqual(sum(r['fractions'].values()),1.)

    def test_production_resamples_cannot_be_reduced(self):
        with self.assertRaisesRegex(ValueError,'99999'):m.analyze_final(prediction_fixture(),[],costs_fixture(),PREREG,repetitions=999)

    def test_complete_P1_simulation_uses_independent_10000_samples_and_proxies(self):
        r=m.power_selection(prediction_fixture('P1'),costs_fixture(m.P1_MODELS,phase='P1'),PREREG)
        self.assertEqual(r['MC_null_draws'],10000);self.assertEqual(r['MC_alternative_draws'],10000)
        self.assertEqual(len(r['selected_R_cells']),180);self.assertEqual(len(r['MDE_conservative_cells']),180)
        self.assertTrue(r['null_and_alternative_samples_independent']);self.assertEqual(r['P1_calendar_amendment']['source_dates'],53)
        self.assertTrue(all(not v['empirical_power_certified'] for v in r['proxy_registry'].values()))
        self.assertIn(r['planning_status'],['target_met_in_simulation','target_not_met_at_R200','unknown_capability_at_R200'])

    def test_P1_missing_date_keeps_calendar_without_poisoning_training_noise(self):
        rows=prediction_fixture('P1')
        for r in rows:
            if r['outcome_day']=='2024-11-15':r.update(eligible=False,y_true='',y_pred='')
        result=m.power_selection(rows,costs_fixture(m.P1_MODELS,phase='P1'),PREREG)
        self.assertEqual(result['P1_calendar_amendment']['source_dates'],53)
        cells=[r for r in result['selected_R_cells'] if r['source']=='training']
        self.assertTrue(all(r['finite_null_draws']==10000 for r in cells))
        self.assertTrue(all(r['finite_alternative_draws']==10000 for r in cells))
        self.assertTrue(all(np.isfinite(v['training_SD_upper95']) for v in result['proxy_registry'].values()))

    def test_ablation_purged53_branch_and_practical_family(self):
        rows=ablation_fixture()
        result=m.compare_matched_ablation_errors(rows,fixture=True,repetitions=999)
        self.assertEqual(len(result['comparisons']),9);self.assertEqual(result['family_alpha'],.05)
        self.assertTrue(all(r['n_unique_origins']==53 for r in result['comparisons']))
        rows[0]['outcome_day']='2024-11-01'
        with self.assertRaises(ValueError):m.compare_matched_ablation_errors(rows,fixture=True,repetitions=999)

    def test_ablation_streaming_requires_presealed_schedule_and_primary_A_closure(self):
        rows=ablation_fixture(production=True)
        self.assertGreater(max(r['checkpoint_closed_at_utc'] for r in rows),min(r['scoring_started_at_utc'] for r in rows))
        result=m.compare_matched_ablation_errors(rows)
        self.assertEqual(result['ablation_schedule_sha256'],HASH)
        self.assertTrue(result['development_streaming_allowed_after_global_primary_A_closure_and_presealed_complete_schedule'])
        for field,value in [('ablation_schedule_closed_at_utc','2026-11-06T14:00:00+00:00'),('all_primary_A_weights_closed_at_utc','2026-11-06T14:00:00+00:00'),('ablation_schedule_sha256','b'*64)]:
            changed=copy.deepcopy(rows);changed[0][field]=value
            with self.assertRaises(ValueError):m.compare_matched_ablation_errors(changed)
        changed=copy.deepcopy(rows);changed[0]['checkpoint_closed_at_utc']=changed[0]['scoring_started_at_utc']
        with self.assertRaisesRegex(ValueError,'own development score'):m.compare_matched_ablation_errors(changed)

    def test_A53_complementary_families_and_no_primary_absence_claim(self):
        rows=inner_fixture(seeds=list(range(680101,680106)));expanded=[]
        for r in rows:
            if r['split']=='validation':continue
            x=dict(r);x.update(source_stage='E2_A53_train_validation',fit_outcome_max='2024-10-31',scaler_fit_outcome_max='2024-10-31',configuration_selection_outcome_max='2024-10-31',scaler_sha256=HASH,config_sha256=HASH,checkpoint_closed_at_utc='2026-11-06T14:00:00+00:00',scoring_started_at_utc='2026-11-06T14:01:00+00:00');expanded.append(x)
        for name in m.MODELS:
            for seed in (['deterministic'] if name in m.DETERMINISTIC else range(680101,680106)):
                for day in m._calendar('2024-11-08','2024-12-30'):
                    expanded.append({'model':name,'seed':seed,'split':'validation','outcome_day':day,'y_true':.02,'y_pred':.021,'eligible':True,'checkpoint_sha256':HASH,'scaler_sha256':HASH,'config_sha256':HASH,'source_stage':'E2_A53_train_validation','fit_outcome_max':'2024-10-31','scaler_fit_outcome_max':'2024-10-31','configuration_selection_outcome_max':'2024-10-31','checkpoint_closed_at_utc':'2026-11-06T14:00:00+00:00','scoring_started_at_utc':'2026-11-06T14:01:00+00:00'})
        result=m.analyze_A53_overfit(expanded)
        self.assertEqual(len(result['violation_family20']),20);self.assertEqual(len(result['tolerance_family20']),20)
        self.assertEqual(result['bootstrap_draws'],19999);self.assertEqual(result['formal_hypotheses'],40)
        self.assertEqual([r['block_days'] for r in result['calendar_block_sensitivities']],[14,28])
        self.assertTrue(all(not r['primary_decision'] for r in result['calendar_block_sensitivities']))
        self.assertTrue(result['does_not_establish_absence_of_overfit_for_primary_Dec24_predictor'])
        self.assertTrue(all(r['family_alpha']==.005 for r in result['tolerance_family20']))
        self.assertTrue(all(not(r['violation_supported'] and r['tolerance_supported']) for r in result['model_summaries']))
        expanded[-1]['configuration_selection_outcome_max']='2024-11-07'
        with self.assertRaises(ValueError):m.analyze_A53_overfit(expanded)

    def test_aggregate_fixture_analysis_end_to_end(self):
        selection={'best_model':'TSMixer','best_baseline':'GRU-E2','outcome_max':'2024-10-31','source':'complete20model_common_development','frozen_before_test':True}
        train=[dict(r,source_stage='E2_primary_train_inference') for r in inner_fixture() if r['split']=='train']
        result=m.analyze_final(prediction_fixture(fixture=True),[],costs_fixture(),PREREG,fixture=True,repetitions=999,efficiency_selection=selection,primary_training_panel=train)
        self.assertEqual(len(result['primary_families']),180)
        self.assertEqual(len(result['diagnostics_199']),199)
        self.assertEqual(len(result['primary_train_test_gap_descriptive']['model_summaries']),20)
        self.assertEqual(len(result['efficiency_equivalence_exploratory']['equivalence19']),19)
        self.assertEqual(result['counts']['models'],20)
        self.assertTrue(all('QLIKE_log_variance_plus_ratio' in r for r in result['seed_metrics']))
        self.assertTrue(all('secondary_metric_temporal_CIs' in r for r in result['aggregate_metrics']))
        self.assertEqual(result['primary_train_test_gap_descriptive']['formal_hypotheses'],0)
        self.assertNotIn('y_true',json.dumps(result));self.assertNotIn('y_pred',json.dumps(result));self.assertEqual(result['fits_performed'],0)
        unknown=[r for r in result['cost_Pareto_axes'] if r['axis']=='FLOPs_per_window'];self.assertTrue(all(r['nondominated'] is None for r in unknown))


if __name__=='__main__':unittest.main()
