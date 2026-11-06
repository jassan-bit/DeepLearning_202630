"""Synthetic diagnostic checks and optional actual-backbone ablation routing.

No market data, test labels or model fitting are used. Set
E2_FINAL_ATTENTION_SOURCES and E2_FINAL_MLP_ROOTS_JSON for all10 integration.
"""
from __future__ import annotations

from datetime import date,timedelta
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e2_final_diagnostics import (TOP10,SEVERITIES,DESIGN,fit_perturbation_reference,
    perturb_raw_windows,evaluate_robustness,torch_normalized_predictor,
    torch_raw_predictor,occlusion_attributions,integrated_gradients,
    deletion_insertion_fidelity,explanation_stability,summarize_daily_values,
    explain_clipped_linear,explain_naive_lag,causal_permutation_importance,
    ablation_definitions,apply_ablation,public_diagnostic_summary,get_ablation_definitions,
    summarize_fidelity,attribution_summary,fit_explanation_baseline,validate_explanation_baseline,FEATURE_GROUPS,
    preprocessing_fingerprint,perturb_target_history,evaluate_endogenous_robustness,
    explain_endogenous_history,evaluate_final_diagnostics)


def fixture(n=14,history=7,first=date(2024,1,1)):
    number=n+history
    close=100+np.arange(number)*.3
    opening=close+np.sin(np.arange(number))*.2
    bars=np.stack([opening,np.maximum(opening,close)+1,np.minimum(opening,close)-1,close,10+np.arange(number)],axis=-1)
    windows=np.stack([bars[i:i+history] for i in range(n)])
    origins=[(first+timedelta(days=i)).isoformat() for i in range(n)]
    return windows,origins


def reference():
    x,days=fixture(20,first=date(2023,1,10))
    return fit_perturbation_reference(x,training_outcome_dates=days,maximum_training_outcome='2023-12-24')


class PositiveLinear(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.arange(1,36,dtype=torch.float32)/35)
    def forward(self,x):return (x.reshape(len(x),35)*self.weight).sum(1)+2


class CorruptionAndRobustnessTests(unittest.TestCase):
    def setUp(self):self.x,self.days=fixture();self.ref=reference()

    def test_three_levels_per_perturbation_and_no_fit_on_import(self):
        self.assertEqual(set(SEVERITIES),{'noise','missing','outliers'})
        self.assertTrue(all(len(v)==3 for v in SEVERITIES.values()))
        self.assertEqual(DESIGN['real_data_fits_performed_by_definition'],0)

    def test_calendar_shared_bar_receives_identical_price_and_volume_shock(self):
        noise,receipt=perturb_raw_windows(self.x,self.days,kind='noise',severity=.05,reference=self.ref)
        np.testing.assert_array_equal(noise[0,1:],noise[1,:-1])
        ratios=noise[...,:4]/self.x[...,:4]
        np.testing.assert_allclose(ratios,np.broadcast_to(ratios[...,:1],ratios.shape),rtol=1e-14)
        self.assertTrue((noise[...,1]>=np.maximum(noise[...,0],noise[...,3])).all())
        self.assertTrue((noise[...,2]<=np.minimum(noise[...,0],noise[...,3])).all())
        self.assertTrue(receipt['calendar_consistent'])
        self.assertFalse(receipt['targets_changed'])

    def test_same_day_draw_is_independent_of_other_windows_in_panel(self):
        whole,_=perturb_raw_windows(self.x,self.days,kind='noise',severity=.03,reference=self.ref)
        sub,_=perturb_raw_windows(self.x[3:9],self.days[3:9],kind='noise',severity=.03,reference=self.ref)
        np.testing.assert_array_equal(whole[3:9],sub)

    def test_noise_levels_couple_same_gaussian_draw(self):
        a,_=perturb_raw_windows(self.x,self.days,kind='noise',severity=.01,reference=self.ref)
        b,_=perturb_raw_windows(self.x,self.days,kind='noise',severity=.03,reference=self.ref)
        np.testing.assert_allclose(np.log(b[...,:4]/self.x[...,:4]),3*np.log(a[...,:4]/self.x[...,:4]),rtol=1e-12,atol=1e-15)

    def test_missing_last_close_uses_raw_training_mean_not_hidden_anchor(self):
        out,_=perturb_raw_windows(self.x,self.days,kind='missing',severity=1,reference=self.ref)
        np.testing.assert_allclose(out,np.broadcast_to(self.ref['raw_OHLCV_mean'],out.shape))
        self.assertNotEqual(out[0,-1,3],self.x[0,-1,3])
        self.assertFalse(np.isnan(out).any())

    def test_outlier_mask_repeats_across_severities_and_preserves_OHLCV(self):
        a,_=perturb_raw_windows(self.x,self.days,kind='outliers',severity=.1,reference=self.ref)
        b,_=perturb_raw_windows(self.x,self.days,kind='outliers',severity=.5,reference=self.ref)
        np.testing.assert_array_equal(a!=self.x,b!=self.x)
        self.assertTrue((b[...,1]>=np.maximum(b[...,0],b[...,3])).all())
        self.assertTrue((b[...,4]>=0).all())

    def test_future_imputation_reference_and_inconsistent_bar_rejected(self):
        bad={**self.ref,'training_outcome_max':'2023-12-31'}
        with self.assertRaises(ValueError):perturb_raw_windows(self.x,self.days,kind='missing',severity=.1,reference=bad)
        badx=self.x.copy();badx[1,0,:4]*=2
        with self.assertRaises(ValueError):perturb_raw_windows(badx,self.days,kind='noise',severity=.1,reference=self.ref)

    def test_calendar_gaps_and_less_than2000_resamples_rejected(self):
        with self.assertRaises(ValueError):perturb_raw_windows(self.x[[0,2]], [self.days[0],self.days[2]],kind='noise',severity=.01,reference=self.ref)
        y=np.ones(14)*.1
        with self.assertRaises(ValueError):evaluate_robustness(lambda x:np.ones(len(x))*.1,self.x,y,self.days,reference=self.ref,repetitions=1999)

    def test_clean_RMSE_is_mean_per_seed_not_ensemble(self):
        y=np.ones(14)*.1
        result=evaluate_robustness(lambda x:np.stack([np.ones(len(x))*.08,np.ones(len(x))*.12]),self.x,y,self.days,reference=self.ref)
        self.assertEqual(result['seed_members'],2)
        self.assertAlmostEqual(result['rows'][0]['clean_RMSE_mean_seed'],.02)
        self.assertTrue(all(r['paired_difference_interval']['requested_draws']==2000 for r in result['rows']))
        self.assertEqual(result['new_hypothesis_decisions'],0)
        self.assertTrue(all(a['relative_degradation_AUC']==0 for a in result['AUC']))
        self.assertNotIn('private_predictions',result)

    def test_zero_clean_error_reports_relative_scores_not_estimable(self):
        y=np.ones(14)*.1
        result=evaluate_robustness(lambda x:np.ones(len(x))*.1,self.x,y,self.days,reference=self.ref)
        self.assertTrue(all(r['relative_degradation'] is None for r in result['rows']))
        self.assertTrue(all(r['relative_degradation_AUC'] is None for r in result['AUC']))
        json.dumps(public_diagnostic_summary(result),allow_nan=False)

    def test_ineligible_dates_are_preserved_not_compacted(self):
        eligible=np.ones(14,bool);eligible[3]=False
        x=self.x.copy();x[3]=np.nan;y=np.ones(14)*.1;y[3]=np.nan
        result=evaluate_robustness(lambda x:np.ones(len(x))*.11,x,y,self.days,reference=self.ref,eligible=eligible)
        self.assertEqual(result['calendar_origins'],14);self.assertEqual(result['eligible_unique_origins'],13)
        with self.assertRaises(ValueError):evaluate_robustness(lambda x:np.ones(len(x))*.1,x,np.ones(14),self.days,reference=self.ref,eligible=eligible)

    def test_string_false_eligibility_does_not_become_true(self):
        with self.assertRaises(ValueError):perturb_raw_windows(self.x,self.days,kind='noise',severity=.01,reference=self.ref,eligible=['False']*14)

    def test_endogenous_state_without_callback_is_NA_not_robustness_win(self):
        r=evaluate_robustness(None,None,None,None,reference=None,input_contract='ARIMA_frozen_endogenous_state')
        self.assertEqual(r['status'],'not_applicable_distinct_information_set')
        self.assertFalse(r['architectural_immunity_claimed'])

    def test_training_reference_must_use_own_distinct_past_outcomes(self):
        with self.assertRaises(ValueError):fit_perturbation_reference(self.x,training_outcome_dates=self.days,maximum_training_outcome='2023-12-24')
        days=[self.days[0]]*len(self.x)
        with self.assertRaises(ValueError):fit_perturbation_reference(self.x,training_outcome_dates=days,maximum_training_outcome='2024-12-24')


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):self.model=PositiveLinear();self.x=np.ones((7,7,5),np.float32)*.5;self.pred=torch_normalized_predictor(self.model)

    def test_occlusion_matches_linear_contributions_and_leaves_weights_unchanged(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        r=occlusion_attributions(self.pred,self.x)
        full=(self.x*self.model.weight.detach().numpy().reshape(1,7,5)).reshape(len(self.x),35)
        expected=np.stack([full[:,g['indices']].sum(1) for g in FEATURE_GROUPS],axis=1)
        np.testing.assert_allclose(r['private_attributions'],expected,rtol=3e-5,atol=2e-6)
        self.assertTrue(r['primary']);self.assertFalse(r['causal_effect_claimed'])
        for k,v in self.model.state_dict().items():torch.testing.assert_close(v,before[k])
        self.assertTrue(self.model.training)

    def test_IG_linear_completeness_and_no_parameter_grad_accumulation(self):
        r=integrated_gradients(self.model,self.x,steps=32)
        self.assertEqual(r['status'],'complete_for_all_rows')
        self.assertEqual(r['completeness_pass_fraction'],1.)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(self.model.training)

    def test_IG_stop_gradient_output_is_not_accepted(self):
        class Stopped(nn.Module):
            def forward(self,x):return x.mean((1,2)).detach()+2
        r=integrated_gradients(Stopped(),self.x,steps=64)
        self.assertEqual(r['completeness_pass_fraction'],0.)
        self.assertEqual(r['status'],'incomplete_not_accepted_as_primary_explanation')
        self.assertIn('occlusion',r['primary_alternative'])

    def test_fidelity_explanation_beats_random_for_monotone_linear_model(self):
        a=occlusion_attributions(self.pred,self.x)['private_attributions']
        r=deletion_insertion_fidelity(self.pred,self.x,a,random_rankings=10)
        self.assertTrue((r['private_deletion_advantage']>0).all())
        self.assertTrue((r['private_insertion_advantage']>0).all())
        self.assertTrue(r['AUC_not_clipped_to_unit_interval'])

    def test_fidelity_near_constant_predictor_is_not_estimable(self):
        r=deletion_insertion_fidelity(lambda x:np.ones(len(x)),self.x,np.ones_like(self.x))
        self.assertEqual(r['not_estimable_near_zero_prediction_difference'],len(self.x))
        self.assertTrue(np.isnan(r['private_deletion_AUC']).all())

    def test_fidelity_rejects_unidentified_ensemble(self):
        with self.assertRaises(ValueError):deletion_insertion_fidelity(lambda x:np.ones((2,len(x))),self.x,np.ones_like(self.x))

    def test_first_five_seed_stability_ignores_extra_seed_and_retains_calendar(self):
        a=np.broadcast_to(np.arange(35).reshape(1,7,5),self.x.shape).copy()
        attrs={i:a.copy() for i in range(1,7)};attrs[6]=-a[:,::-1].copy()
        days=[(date(2024,1,1)+timedelta(days=i)).isoformat() for i in range(7)]
        r=explanation_stability(attrs,[1,2,3,4,5,6],days)
        self.assertEqual(r['prespecified_first_five_seeds'],[1,2,3,4,5]);self.assertEqual(r['pair_count'],10)
        self.assertAlmostEqual(r['overall']['mean'],1.)
        self.assertTrue(r['overall']['calendar_not_compacted'])

    def test_constant_attribution_maps_are_not_pretended_unstable_zero(self):
        attrs={i:np.ones((len(self.x),14)) for i in range(5)}
        days=[(date(2024,1,1)+timedelta(days=i)).isoformat() for i in range(7)]
        r=explanation_stability(attrs,list(range(5)),days)
        self.assertEqual(r['overall']['status'],'not_estimable');self.assertIsNone(r['overall']['mean'])

    def test_clipped_linear_analytic_explanation_handles_crossing_zero(self):
        x=np.zeros((2,7,5));x[0,0,0]=2;x[1,0,0]=.5
        coefficient=np.zeros(35);coefficient[0]=1
        r=explain_clipped_linear(coefficient,-1,x)
        self.assertLess(r['completeness_max_absolute_residual'],1e-12)
        np.testing.assert_allclose(r['private_prediction_difference'],[1,0])

    def test_naive_explanation_uses_only_named_observed_lag(self):
        history=np.arange(1,29,dtype=float).reshape(2,14)/1000
        r=explain_naive_lag(history,lag=7,training_mean=.01)
        self.assertTrue(np.flatnonzero(r['private_attributions'][0]).tolist()==[7])
        np.testing.assert_array_equal(r['private_prediction'],history[:,-7])

    def test_public_export_strips_individual_explanations(self):
        r=public_diagnostic_summary(occlusion_attributions(self.pred,self.x))
        self.assertFalse(any(k.startswith('private_') for k in r))
        json.dumps(r,allow_nan=False)

    def test_public_fidelity_summary_retains_quantitative_paired_intervals(self):
        a=occlusion_attributions(self.pred,self.x)['private_attributions']
        fidelity=deletion_insertion_fidelity(self.pred,self.x,a)
        days=[(date(2024,1,1)+timedelta(days=i)).isoformat() for i in range(7)]
        summary=summarize_fidelity(fidelity,days)
        self.assertGreater(summary['metric_summaries']['deletion_advantage']['mean'],0)
        self.assertEqual(summary['metric_summaries']['insertion_advantage']['requested_draws'],2000)
        json.dumps(public_diagnostic_summary(summary),allow_nan=False)

    def test_attribution_summary_reports35features_and_normalized_channel_lag_shares(self):
        a=occlusion_attributions(self.pred,self.x)['private_attributions']
        days=[(date(2024,1,1)+timedelta(days=i)).isoformat() for i in range(7)]
        summary=attribution_summary(a,days)
        self.assertEqual(len(summary['groups']),14)
        self.assertAlmostEqual(sum(r['mean'] for r in summary['channels']),1.)
        self.assertAlmostEqual(sum(r['mean'] for r in summary['lags']),1.)
        self.assertFalse(summary['causal_effect_claimed'])

    def test_training_cellwise_baseline_preserves_deterministic_anchor_without_test_data(self):
        x=np.arange(10*7*5,dtype=float).reshape(10,7,5);x[:,-1,3]=.143
        days=[(date(2023,1,1)+timedelta(days=i)).isoformat() for i in range(10)]
        result=fit_explanation_baseline(x,training_outcome_dates=days,maximum_training_outcome='2023-12-24',scaler_sha256='a'*64)
        self.assertTrue(result['constant_training_cells'][-1][3])
        self.assertAlmostEqual(result['baseline'][-1][3],.143)
        self.assertFalse(result['test_or_evaluation_data_used'])
        validate_explanation_baseline(result,expected_scaler_sha256='a'*64,maximum_training_outcome='2023-12-24')
        with self.assertRaises(ValueError):validate_explanation_baseline(result,expected_scaler_sha256='b'*64,maximum_training_outcome='2023-12-24')
        with self.assertRaises(ValueError):fit_explanation_baseline(x,training_outcome_dates=days,maximum_training_outcome='2022-12-24',scaler_sha256='a'*64)

    def test_name_based_ablation_registry_has30_exact_independent_definitions(self):
        self.assertEqual(sum(len(get_ablation_definitions(name)) for name in TOP10),30)
        first=get_ablation_definitions('TSMixer');first['without_moments']['actions'].clear()
        self.assertTrue(get_ablation_definitions('TSMixer')['without_moments']['actions'])
        with self.assertRaises(ValueError):get_ablation_definitions('Ridge')

    def test_causal_permutation_never_moves_values_between_origins(self):
        x,days=fixture(7);y=x[:,-1,3]/1000;seen=[]
        def predictor(a):
            for row in a:
                self.assertTrue(any(np.array_equal(np.sort(row,axis=0),np.sort(original,axis=0)) for original in x))
            seen.append(len(a));return a[:,-1,3]/1000+.001
        r=causal_permutation_importance(predictor,x,y,days,groups={'OHLC_order':[0,1,2,3],'volume_order':[4]})
        self.assertEqual(len(r['groups']),2);self.assertFalse(r['future_history_rows_used'])


class EndogenousAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setup_case(self):
        raw,origins=fixture(7)
        history_days=[(date(2023,12,1)+timedelta(days=i)).isoformat() for i in range(45)]
        history={'dates':history_days,'values':.01+np.arange(45)*.0001}
        cutoff='2023-12-24';mean=float(np.mean(history['values'][:24]))
        def callback(lag=None):
            def predict(h,origin):
                self.assertTrue(all(date.fromisoformat(d)<date.fromisoformat(origin) for d in h['dates']))
                self.assertEqual(h['dates'][-1],(date.fromisoformat(origin)-timedelta(days=1)).isoformat())
                if lag:return float(h['values'][h['dates'].index((date.fromisoformat(origin)-timedelta(days=lag)).isoformat())])
                return .8*float(h['values'][-1])+.2*mean
            return predict
        entries={name:{'input_kind':'endogenous','fit_outcome_max':cutoff,'training_sigma_mean':mean,
            'predict_from_history':callback(lag),'lag':lag,'seed_ids':['deterministic']}
            for name,lag in [('Persistence',1),('SeasonalNaive',7),('ARIMA',None)]}
        return raw,origins,history,entries

    def test_endogenous_history_faults_preserve_prefit_and_never_use_future(self):
        raw,origins,h,entries=self.setup_case();entry=entries['ARIMA']
        prefix={'dates':h['dates'][:31],'values':h['values'][:31]}
        for kind in SEVERITIES:
            severity=1 if kind=='missing' else .5
            changed,receipt=perturb_target_history(prefix,kind=kind,severity=severity,
                fit_outcome_max=entry['fit_outcome_max'],training_sigma_mean=entry['training_sigma_mean'])
            np.testing.assert_array_equal(changed['values'][:24],prefix['values'][:24])
            self.assertFalse(receipt['parameters_refitted'])
        y=np.ones(7)*.014
        result=evaluate_endogenous_robustness(entry,h,y,origins)
        self.assertEqual(len(result['rows']),9)
        self.assertFalse(result['architectural_immunity_claimed'])
        self.assertTrue(result['input_faults_are_not_the_same_OHLCV_faults'])

    def test_endogenous_occlusion_is_real_for_all_three_state_or_lag_contracts(self):
        raw,origins,h,entries=self.setup_case()
        for name,entry in entries.items():
            r=explain_endogenous_history(entry,h,origins)
            self.assertTrue(r['past_history_only'])
            self.assertEqual(r['seed_stability']['status'],'not_applicable_deterministic_estimator')
            expected={'Persistence':1,'SeasonalNaive':7,'ARIMA':8}[name]
            self.assertEqual(r['private_attributions'].shape,(7,expected))
            self.assertTrue(np.isfinite(r['private_attributions']).all())
            self.assertFalse(r['coefficient_parameters_refitted'])

    def test_scaler_fingerprint_excludes_receipt_without_hiding_transform_changes(self):
        p={'x_mean':[0]*5,'x_scale':[1]*5,'y_scale':.01}
        a=preprocessing_fingerprint(p)
        self.assertEqual(a,preprocessing_fingerprint({**p,'scaler_sha256':a,'explanation_baseline_receipt':{}}))
        self.assertNotEqual(a,preprocessing_fingerprint({**p,'y_scale':.02}))

    def test_full_twenty_model_pipeline_routes_actual_robustness_and_three_XAI_families(self):
        import tempfile
        from dl_data_v2 import relative_features
        raw,origins,history,entries=self.setup_case();train,train_dates=fixture(20,first=date(2023,1,10))
        pre={'history_days':7,'x_mean':[0]*5,'x_scale':[1]*5,'y_scale':.001,'last_fit_outcome_day_utc':'2023-12-24'}
        fingerprint=preprocessing_fingerprint(pre)
        baseline=fit_explanation_baseline(relative_features(train).astype(np.float32),training_outcome_dates=train_dates,
            maximum_training_outcome='2023-12-24',scaler_sha256=fingerprint)
        models=[PositiveLinear() for _ in range(5)]
        predictors=[torch_raw_predictor(m,pre) for m in models]
        predict5=lambda x:np.stack([p(x) for p in predictors])
        for i in range(16):entries['NN'+str(i)]={'input_kind':'OHLCV','seed_ids':list(range(670101,670106)),'predict_raw':predict5}
        entries['NN0']['members']=[{'model':m,'preprocessing':pre,'seed':seed,'scaler_sha256':fingerprint,'explanation_baseline_receipt':baseline} for m,seed in zip(models,range(670101,670106))]
        coef=np.ones(35)*.0002;intercept=.02
        pn=lambda x:np.maximum(0,x.reshape(len(x),35)@coef+intercept)
        entries['Ridge']={'model':'Ridge','input_kind':'OHLCV','seed_ids':['deterministic'],'predict_raw':lambda x:pn(relative_features(x)),
            'predict_normalized':pn,'preprocessing':pre,'explanation_baseline_receipt':baseline,'scaler_sha256':fingerprint,
            'linear_coefficients':coef,'linear_intercept':intercept}
        self.assertEqual(len(entries),20)
        closure={'all_model_weights_closed':True,'catalogue':list(entries),'scope':'synthetic_fixture_not_a_real_benchmark'}
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            r=evaluate_final_diagnostics(entries,raw,np.ones(7)*.014,origins,best3_development=['NN0','Ridge','ARIMA'],
                reference=reference(),history_source=history,private_output=output/'private',public_output=output/'public',closure_receipt=closure)
            self.assertEqual(r['models_evaluated'],20)
            self.assertEqual(set(r['explanations']),{'NN0','Ridge','ARIMA'})
            self.assertTrue(all(len(v['rows'])==9 for v in r['robustness'].values()))
            self.assertEqual(len(r['explanations']['NN0']['members']),5)
            self.assertEqual(r['explanations']['Ridge']['seed_stability']['status'],'not_applicable_deterministic_estimator')
            self.assertTrue((output/'private/diagnostics_private.json').exists())
            public=(output/'public/summary.json').read_text()
            self.assertNotIn('private_attributions',public);self.assertNotIn('raw_OHLCV_mean',public)
            self.assertEqual(json.loads(public)['real_model_fits_performed'],0)
            with self.assertRaises(ValueError):evaluate_final_diagnostics(entries,raw,np.ones(7)*.014,origins,best3_development=['NN0','Ridge','ARIMA'],
                reference=reference(),history_source=history,private_output=output/'private2',closure_receipt={'all_model_weights_closed':False,'catalogue':list(entries)})


class ActualBackboneAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        attention=os.environ.get('E2_FINAL_ATTENTION_SOURCES');mlp=os.environ.get('E2_FINAL_MLP_ROOTS_JSON')
        if not attention or not mlp:raise unittest.SkipTest('Explicit source roots for all10 final models not supplied')
        from e2_final_models_attention import build_model as attention_build
        from e2_final_models_mlp import build_model as mlp_build
        cls.attention_build=staticmethod(attention_build);cls.mlp_build=staticmethod(mlp_build)
        cls.attention_roots={name:Path(attention)/name for name in TOP10[5:]}
        cls.mlp_roots=json.loads(Path(mlp).read_text()) if Path(mlp).is_file() else json.loads(mlp)
        torch.set_num_threads(1)

    def make(self,name):
        builder,roots=(self.mlp_build,self.mlp_roots) if name in TOP10[:5] else (self.attention_build,self.attention_roots)
        return builder(name,candidate={'dropout':.1},source_roots=roots)

    def test_all30_actual_component_routes_forward_backward_and_retraining_receipts(self):
        records=[]
        for name in TOP10:
            model=self.make(name);definitions=ablation_definitions(model)
            self.assertEqual(len(definitions),3)
            self.assertEqual(definitions,get_ablation_definitions(name))
            for key in definitions:
                with self.subTest(model=name,ablation=key):
                    model.train();model.zero_grad(set_to_none=True)
                    handle=apply_ablation(model,key)
                    self.assertFalse(handle.receipt['retrained_ablation_result_claimed'])
                    self.assertTrue(handle.receipt['operator_must_be_reinstalled_when_loading_ablated_checkpoint'])
                    self.assertGreater(handle.receipt['remaining_trainable_parameters'],1)
                    x=torch.randn(2,7,5);y=model(x)
                    self.assertTrue(torch.isfinite(y).all() and (y>=0).all())
                    y.square().mean().backward()
                    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                    self.assertEqual(missing,[])
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    handle.remove()
                    self.assertFalse(hasattr(model,'_e2_ablation_definition'))
                    records.append((name,key))
        self.assertEqual(len(records),30)

    def test_moment_lesion_keeps_readout_trainable_and_hook_restore_is_exact(self):
        model=self.make('TSMixer');model.eval();x=torch.randn(2,7,5)
        with torch.no_grad():original=model(x)
        model.train();handle=apply_ablation(model,'without_moments');model.eval()
        self.assertTrue(model.readout.weight.requires_grad)
        with torch.no_grad():model(x)
        handle.remove()
        with torch.no_grad():restored=model(x)
        torch.testing.assert_close(original,restored)

    def test_hooks_are_not_silently_stackable_or_post_fit(self):
        model=self.make('S-Mamba');handle=apply_ablation(model,'without_reverse_SSM')
        with self.assertRaises(ValueError):apply_ablation(model,'without_FF_refinement')
        model._e2_fit_started=True
        with self.assertRaises(ValueError):handle.remove()
        fresh=self.make('S-Mamba');fresh._e2_fit_started=True
        with self.assertRaises(ValueError):apply_ablation(fresh,'without_reverse_SSM')


if __name__=='__main__':unittest.main()
