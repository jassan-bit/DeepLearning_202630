"""Fixture-only controls for the new benchmark engine; no market models fitted."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_e2_final_benchmark as engine


class CalendarAndRegistryTests(unittest.TestCase):
    def test_nested_blocks_purged_and_contiguous(self):
        for outer in engine.OUTER_ORIGINS:
            blocks=engine.inner_blocks(outer)
            self.assertEqual(blocks[0]['origin'], outer-timedelta(days=97))
            self.assertEqual(blocks[-1]['end'], outer-timedelta(days=8))
            self.assertEqual([len(engine.calendar(b['origin'], b['end'])) for b in blocks], [30]*3)
            self.assertEqual(blocks[0]['end']+timedelta(days=1), blocks[1]['origin'])

    def test_exact_prospective_cache_counts(self):
        plan=engine.planned_inventory()
        self.assertEqual(len(plan['unique_inner_origins']),22)
        self.assertEqual(len(plan['unique_inner_plus_finalsearch_origins']),32)
        self.assertEqual(plan['cache_unique_HPO_neural_trajectories'],9984)
        self.assertEqual(plan['HPO_optimizer_fit_upper_bound_with_exact_caching'],19968)
        self.assertEqual(plan['HPO_estimator_fit_upper_bound_with_exact_caching'],768)
        self.assertEqual(plan['anchor_optimizer_fits'],120)
        self.assertEqual(plan['all_tunable_assigned_CPU_seconds'],2592000)

    def test_plan_default_never_collects_or_loads_data(self):
        with patch.object(engine,'collect_inputs',side_effect=AssertionError('opened inputs')), \
             patch.object(engine,'development',side_effect=AssertionError('opened market')), redirect_stdout(io.StringIO()):
            result=engine.main([])
        self.assertEqual(result['dataset_payloads_opened'],0)

    def test_source_points_and_undeclared_exception(self):
        registry=engine.read_json(ROOT/'experiments/e2_completion_20261106/candidate_registry.json')
        hpo=engine.read_json(ROOT/'experiments/e2_completion_20261106/hpo.json')
        engine.validate_candidates(registry,hpo)
        for model in engine.TUNABLE_NEURAL:
            self.assertEqual(sum(r.get('source_reference_point') is True for r in registry if r['model']==model),1)
        bad=deepcopy(registry)
        row=next(r for r in bad if r['model']=='TSMixer' and not r.get('source_reference_point'))
        row['parameters']['dropout']=.9
        with self.assertRaises(ValueError):engine.validate_candidates(bad,hpo)
        bad=deepcopy(registry)
        next(r for r in bad if r.get('source_reference_point'))['parameters']['weight_decay']=.2
        with self.assertRaises(ValueError):engine.validate_candidates(bad,hpo)

    def test_split_preserves_missing_calendar_and_purge(self):
        dates=engine.calendar('2023-01-01','2024-12-31')
        eligible=np.arange(len(dates))>=8
        eligible[82]=False
        data={'calendar':dates,'eligible_by_history':{7:eligible}}
        split=engine.split_indices(data,'2024-11-08')
        self.assertNotIn(82, split['train'])
        self.assertEqual(dates[max(split['train'])],date(2024,10,31))
        self.assertGreaterEqual((dates[min(split['inner_validation'])]-dates[max(split['inner_train'])]).days,8)
        self.assertEqual(len(engine.calendar(engine.P1_ORIGIN,engine.P1_END)),53)

    def test_missing_evaluation_date_is_not_silently_removed(self):
        dates=engine.calendar('2024-01-01','2024-01-31');mask=np.ones(31,bool);mask[9]=False
        with self.assertRaises(ValueError):engine.evaluation_indices({'calendar':dates,'eligible_by_history':{7:mask}},dates[0],dates[-1])

    def test_job_cannot_invent_a_candidate_or_seed(self):
        registry=engine.read_json(ROOT/'experiments/e2_completion_20261106/candidate_registry.json')
        candidate=next(r for r in registry if r['model']=='Ridge')
        context={'candidates':registry}
        engine.validate_requested_fit(context,'Ridge',candidate,'deterministic','2024-01-01','HPO')
        with self.assertRaises(ValueError):engine.validate_requested_fit(context,'Ridge',candidate,123,'2024-01-01','HPO')
        invented=deepcopy(candidate);invented['parameters']['alpha']=.123456
        with self.assertRaises(ValueError):engine.validate_requested_fit(context,'Ridge',invented,'deterministic','2024-01-01','HPO')
        with self.assertRaises(ValueError):engine.validate_requested_fit(context,'Ridge',candidate,'deterministic','2026-09-01','HPO')


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)/'private';self.base.mkdir()
        self.inputs={'source_files':{'fixture':'a'*64},'development_sources':{'fixture':'b'*64}}
        self.hash=engine.hash_json(self.inputs)
        self.context={'output':self.base,'inputs':self.inputs}
    def tearDown(self):self.tmp.cleanup()

    def test_private_output_and_relative_source_escape(self):
        with self.assertRaises(ValueError):engine.private_output(ROOT/'results/new')
        with self.assertRaises(ValueError):engine.local(self.base,'../outside')
        with self.assertRaises(ValueError):engine.local(self.base,'/etc/passwd')

    def test_phase_artifact_tamper_fails(self):
        phase=self.base/'phase';phase.mkdir();artifact=phase/'prediction.csv';artifact.write_text('fixture\n')
        engine.close_phase(phase/'completion.json',self.hash,artifacts=[artifact])
        engine.phase_receipt(phase/'completion.json',expected_inputs=self.hash)
        artifact.write_text('changed\n')
        with self.assertRaises(ValueError):engine.phase_receipt(phase/'completion.json',expected_inputs=self.hash)

    def test_cross_phase_requires_registered_root(self):
        phase=self.base/'phase';phase.mkdir();other=self.base/'other.json';other.write_text('{}')
        with self.assertRaises(ValueError):engine.close_phase(phase/'completion.json',self.hash,artifacts=[other])
        engine.write_json(self.base/'registration.json',{})
        engine.close_phase(phase/'completion.json',self.hash,artifacts=[other])
        engine.phase_receipt(phase/'completion.json',expected_inputs=self.hash)
        with self.assertRaises(ValueError):engine.phase_receipt(phase/'completion.json',expected_inputs='wrong')

    def make_fit(self):
        folder=self.base/'fits/TSMixer/TSMixer-c01/2024-01-01/seed_2026110611';folder.mkdir(parents=True)
        (folder/'checkpoint.pt').write_bytes(b'fixture weights only')
        engine.write_json(folder/'setup.json',{'fixture':True})
        record={'status':'passed','registration_inputs_sha256':self.hash,'identity':{'seed':2026110611},
                'budget_scope':'HPO','artifacts':{name:engine.digest(folder/name) for name in ('checkpoint.pt','setup.json')}}
        engine.write_json(folder/'completion.json',record)
        score=self.base/'development_scores/fixed';score.mkdir(parents=True)
        (score/'predictions.csv').write_text('model,y_true,y_pred\nfixture,1,1\n')
        engine.close_phase(score/'completion.json',self.hash,artifacts=[score/'predictions.csv'])
        return folder,score/'completion.json'

    def test_pruned_fit_requires_real_immutable_score_chain(self):
        folder,score=self.make_fit()
        engine.prune_development_fit(self.context,folder,score)
        self.assertFalse((folder/'checkpoint.pt').exists())
        engine.verify_job(folder,self.hash)
        (score.parent/'predictions.csv').write_text('forged\n')
        with self.assertRaises(ValueError):engine.verify_job(folder,self.hash)

    def test_missing_weights_without_pruning_receipt_fail(self):
        folder,_=self.make_fit();(folder/'checkpoint.pt').unlink()
        with self.assertRaises(ValueError):engine.verify_job(folder,self.hash)

    def test_forged_pruning_cannot_authorize_missing_weights(self):
        folder,score=self.make_fit();engine.prune_development_fit(self.context,folder,score)
        prune=folder/'checkpoint_pruned_after_development_scoring.json';value=engine.read_json(prune)
        value['original_fit_receipt_sha256']='0'*64;engine.write_json(prune,value,exclusive=False)
        with self.assertRaises(ValueError):engine.verify_job(folder,self.hash)

    def test_no_new_fit_after_final_scoring_started(self):
        (self.base/'final_scoring').mkdir()
        engine.write_json(self.base/'final_scoring/started.json', {'synthetic_fixture':True})
        candidates=engine.read_json(ROOT/'experiments/e2_completion_20261106/candidate_registry.json')
        candidate=next(c for c in candidates if c['model']=='Ridge')
        context={**self.context,'candidates':candidates}
        with patch.object(engine,'check_registration'), \
             patch.object(engine,'development',side_effect=AssertionError('should not read any data')):
            with self.assertRaisesRegex(ValueError,'after the single final scoring'):
                engine.fit_trial(context,'Ridge',candidate,'deterministic','2024-01-01',budget_scope='HPO')
        self.assertFalse((self.base/'fits').exists())

    def test_oof_score_key_uses_actual_inherited_selection_date(self):
        model='GRU/D7-c03';candidate=engine.fixed_candidate(model);origin=date(2024,1,1)
        folder=engine.job_folder(self.base,model,candidate,engine.HPO_SEEDS[0],origin)
        folder.mkdir(parents=True)
        engine.write_json(folder/'completion.json', {'status':'passed',
            'registration_inputs_sha256':self.hash,'artifacts':{}})
        key=engine.hash_json({'fit_sha256':engine.digest(folder/'completion.json'),
            'start':origin,'end':engine.month_end(origin),'stage':'E2_nested_development',
            'configuration_selection_outcome_max':date(2024,10,31)})
        score=self.base/'development_scores'/key;score.mkdir(parents=True)
        (score/'predictions.csv').write_text('fixture\n')
        engine.close_phase(score/'completion.json',self.hash,artifacts=[score/'predictions.csv'])
        records=engine.oof_fit_records(self.context,[{'run_id':'run-fixture',
            'fit_directory':folder.relative_to(self.base).as_posix(),'fit_origin':origin.isoformat(),
            'configuration_selection_outcome_max':'2024-10-31'}])
        self.assertEqual(records['run-fixture']['scoring_receipt'],
                         (score/'completion.json').relative_to(self.base).as_posix())

    def test_registration_rejects_runtime_drift(self):
        engine.write_json(self.base/'registration.json',{'immutable_inputs':self.inputs,
            'registration_inputs_sha256':self.hash,
            'environment':{'python':'fixture-original','system':'fixture','packages':{}}})
        with self.assertRaisesRegex(ValueError,'runtime or package versions changed'):
            engine.check_registration(self.context)

    def test_ablation_schedule_is_sealed_before_streaming_fits(self):
        model='TimeMixer';seed_ids=list(range(670101,670106))
        engine.write_json(self.base/'registration.json',{})
        engine.write_json(self.base/'all_final_weights/completion.json',{
            'status':'passed','registration_inputs_sha256':self.hash,'artifacts':{},
            'all_primary_and_A_weights_closed_at_utc':'2025-01-01T00:00:00+00:00'})
        engine.write_json(self.base/'hpo/efficiency_selection.json',{'best_top10':model})
        engine.write_json(self.base/'hpo/primary_recipes.json',{model:{'candidate':{
            'model':model,'candidate_id':model+'-c01','parameters':{'dropout':0.}},
            'selection_outcome_max':'2024-10-31'}})
        engine.write_json(self.base/'power/selection.json',{'selected_R':5})
        engine.write_json(self.base/'preflight/ablation_definitions.json',
                          {model:{'lesion1':{},'lesion2':{},'lesion3':{}}})
        context={**self.context,'prereg':{'power':{'final_seed_ids_at_cap':seed_ids}}}
        state={'clock':datetime(2026,1,1,tzinfo=timezone.utc),'fit_count':0}
        def clock():
            state['clock']+=timedelta(seconds=1)
            return state['clock'].isoformat()
        def run(jobs,**kwargs):
            name,candidate,seed,origin=jobs[0];folder=engine.job_folder(self.base,name,candidate,seed,origin)
            self.assertTrue((self.base/'ablation/schedule.json').is_file())
            engine.write_json(folder/'started.json',{'started_at_utc':clock()})
            state['fit_count']+=1
            return [{'record':{'status':'passed'}}]
        def scored(context,name,candidate,seed,origin,start,end,**kwargs):
            return [{'model':name,'seed':seed,'outcome_day':d.isoformat(),
                     'scoring_started_at_utc':clock()} for d in engine.calendar(start,end)]
        runner=SimpleNamespace(run=run)
        analyzed=[]
        def analyze(rows):
            analyzed.extend(rows)
            return {'synthetic_fixture_only':True}
        with patch.object(engine,'stamp',side_effect=clock), \
             patch.object(engine,'TrialRunner',return_value=runner), \
             patch.object(engine,'scored_rows',side_effect=scored), \
             patch('e2_final_statistics.compare_matched_ablation_errors',side_effect=analyze):
            result=engine.run_ablation(context,SimpleNamespace())
        self.assertEqual(state['fit_count'],20);self.assertEqual(len(analyzed),20*53)
        self.assertEqual(result['development_days'],53)
        self.assertEqual(len({r['ablation_schedule_sha256'] for r in analyzed}),1)
        self.assertTrue(all(r['ablation_schedule_closed_at_utc']<r['ablation_fit_started_at_utc']
                            <r['scoring_started_at_utc'] for r in analyzed))


class DevelopmentPanelTests(unittest.TestCase):
    def rows(self,model='Ridge'):
        rows=[]
        for d in engine.calendar('2024-01-01','2024-10-31'):
            cutoff=d.replace(day=1)-timedelta(days=8)
            rows.append({'model':model,'seed':'deterministic','outcome_day':d.isoformat(),
                         'source_stage':'E2_nested_development','eligible':True,'y_true':.02,'y_pred':.018,
                         'fit_outcome_max':cutoff.isoformat(),'scaler_fit_outcome_max':cutoff.isoformat(),
                         'configuration_selection_outcome_max':cutoff.isoformat(),
                         'run_id':d.strftime('%Y%m'),'checkpoint_sha256':d.strftime('%Y%m')})
        return rows
    def test_complete_single_deterministic_panel_passes(self):
        result=engine.validate_development_panel(self.rows(),models=('Ridge',))
        self.assertEqual(result['rows'],305)
    def test_missing_future_or_relabelled_p0_rejected(self):
        for mutation in ('missing','future','P0','checkpoint','duplicate'):
            with self.subTest(mutation=mutation):
                rows=self.rows()
                if mutation=='missing':rows.pop()
                if mutation=='future':rows[0]['configuration_selection_outcome_max']='2024-12-30'
                if mutation=='P0':rows[0]['source_stage']='P0'
                if mutation=='checkpoint':rows[1]['checkpoint_sha256']='changed'
                if mutation=='duplicate':rows.append(rows[0])
                with self.assertRaises(ValueError):engine.validate_development_panel(rows,models=('Ridge',))
    def test_legacy_anchors_require_truthful_historical_exposure(self):
        model='GRU/D7-c03';rows=[]
        for seed in engine.HPO_SEEDS:
            for row in self.rows(model):
                row.update(seed=seed,candidate_id=engine.fixed_candidate(model)['candidate_id'],
                           configuration_selection_outcome_max='2024-10-31',
                           epoch_selection_outcome_max=row['fit_outcome_max'],
                           historical_selection_exposure_disclosed=True,
                           configuration_selection_kind='inherited_fixed_E1_recipe',
                           E2_new_configuration_selection=False,unbiased_nested_selection_claimed=False)
                rows.append(row)
        engine.validate_development_panel(rows,models=(model,))
        rows[0]['configuration_selection_outcome_max']='2023-12-24'
        with self.assertRaises(ValueError):engine.validate_development_panel(rows,models=(model,))


class EstimatorTests(unittest.TestCase):
    def test_ridge_matches_primal_closed_form_with_unpenalized_intercept(self):
        rng=np.random.default_rng(50);x=rng.normal(size=(19,7,5));y=rng.normal(size=19);alpha=.7
        fit=engine.ridge_fit(x,y,alpha);flat=x.reshape(19,-1);center=flat-flat.mean(axis=0)
        expected=np.linalg.solve(center.T@center+alpha*np.eye(35),center.T@(y-y.mean()))
        np.testing.assert_allclose(fit['coef'],expected,rtol=1e-11,atol=1e-11)
        self.assertAlmostEqual(fit['intercept'],y.mean()-flat.mean(axis=0)@expected)
    def test_fresh_tiny_fit_is_reproducible_and_honors_source_no_decay(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(35,1);self.bias=torch.nn.Parameter(torch.tensor(.1));self.bias._no_weight_decay=True
            def forward(self,x):return torch.nn.functional.softplus(self.linear(x.reshape(len(x),-1))[:,0]+self.bias)
        x=np.random.default_rng(4).normal(size=(12,7,5)).astype('float32');y=np.full(12,.6,dtype='float32')
        candidate={'parameters':{'learning_rate':.001,'weight_decay':.01,'batch_size':4}}
        settings={'maximum_epochs':3,'patience':1,'min_delta':0.,'maximum_optimizer_fit_seconds':20,'gradient_clip_norm':1.}
        with patch.object(engine,'build_neural',side_effect=lambda *a:Tiny()):
            one=engine.train_neural('fixture',candidate,{},x,y,42,settings,x_validation=x[:4],y_validation=y[:4])
            two=engine.train_neural('fixture',candidate,{},x,y,42,settings,x_validation=x[:4],y_validation=y[:4])
        np.testing.assert_array_equal(one['training_prediction_scaled'],two['training_prediction_scaled'])
        self.assertEqual(one['optimizer_groups']['source_no_weight_decay_parameters'],1)
        self.assertTrue(one['model']._e2_fit_started)
        self.assertGreaterEqual(one['selected_epoch'],1)
        self.assertLessEqual(one['selected_epoch'],3)

    def test_negative_controls_fit_distinct_weights_with_permuted_stopping_labels(self):
        import pandas as pd
        import torch
        from dl_data import make_windows
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(35,1)
            def forward(self,x):
                return torch.nn.functional.softplus(self.linear(x.reshape(len(x),-1))[:,0])
        dates=pd.date_range('2023-01-01','2024-12-31',tz='UTC');n=len(dates)
        rng=np.random.default_rng(230)
        prices=100*np.exp(np.cumsum(rng.normal(0,.01,n)))
        features=np.column_stack([prices,prices*1.01,prices*.99,prices,rng.uniform(10,50,n)])
        targets=.02+.002*np.sin(np.arange(n)/9);targets[:7]=np.nan
        windows=make_windows(features,7);eligible=np.isfinite(windows).all(axis=(1,2))&np.isfinite(targets)
        data={'calendar':dates,'features':features,'targets':targets,
              'windows_by_history':{7:windows},'eligible_by_history':{7:eligible},
              'persistence':np.concatenate(([np.nan],targets[:-1]))}
        registry=engine.read_json(ROOT/'experiments/e2_completion_20261106/candidate_registry.json')
        recipes={m:{'candidate':next(c for c in registry if c['model']==m)} for m in ['GRU-E2','Ridge']}
        optimization={'maximum_epochs':2,'patience':1,'min_delta':0.,
                      'maximum_optimizer_fit_seconds':20,'gradient_clip_norm':1.}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            engine.write_json(output/'hpo/primary_recipes.json',recipes)
            context={'output':output,'root':ROOT,'inputs':{'development_sources':{'fixture':'a'*64}},
                     'roots':{},'candidates':registry,'hpo':{'optimization':optimization}}
            with patch.object(engine,'check_registration'),patch.object(engine,'development',return_value=data), \
                 patch.object(engine,'build_neural',side_effect=lambda *args:Tiny()):
                for name in ['GRU-E2','Ridge']:
                    candidate=engine.shuffled_control_candidate(recipes[name]['candidate'])
                    seed='deterministic' if name=='Ridge' else engine.NEGATIVE_CONTROL_SEED
                    result=engine.fit_trial(context,name,candidate,seed,'2025-01-01',budget_scope='negative_controls')
                    self.assertEqual(result['status'],'passed',result.get('error'))
                    folder=engine.job_folder(output,name,candidate,seed,'2025-01-01')
                    setup=engine.read_json(folder/'setup.json')
                    self.assertTrue(setup['control_training_labels_are_permuted'])
                    expected=['outer'] if name=='Ridge' else ['outer','inner_train','inner_validation']
                    self.assertEqual(set(setup['label_permutation']),set(expected))
                    for part in expected:
                        receipt=setup['label_permutation'][part]
                        self.assertTrue(receipt['nonidentity_index_permutation'])
                        self.assertTrue(receipt['targets_multiset_preserved'])
                        self.assertLessEqual(receipt['training_outcome_max'],'2024-12-24')
                    self.assertEqual(setup['negative_control'],True)
                    self.assertEqual(result['optimizer_fits'],0 if name=='Ridge' else 2)
                    self.assertNotEqual(folder,engine.job_folder(output,name,recipes[name]['candidate'],seed,'2025-01-01'))


class TrainingDiagnosticTests(unittest.TestCase):
    def test_all_models_share_structurally_available_training_dates(self):
        import pandas as pd
        calendar=pd.date_range('2023-01-01','2023-03-10',tz='UTC');n=len(calendar)
        targets=.02+.003*np.sin(np.arange(n));targets[:7]=np.nan;targets[35]=np.nan
        indices=np.flatnonzero(np.isfinite(targets))
        data={'calendar':calendar,'targets':targets,'eligible_by_history':{7:np.isfinite(targets)}}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);members=[]
            for model in ['GRU-E2','Persistence','SeasonalNaive']:
                folder=output/model;folder.mkdir()
                prediction=targets[indices]+.001
                if model in engine.NAIVES:
                    lag=1 if model=='Persistence' else 7
                    prediction=np.asarray([targets[i-lag] if i>=lag else np.nan for i in indices])
                np.savez(folder/'training_evaluation.npz',train_indices=indices,
                         train_truth=targets[indices],train_prediction=prediction)
                engine.write_json(folder/'setup.json',{'scaler_fit_outcome_max':'2023-03-10',
                                  'training_inference_started_at_utc':'2026-01-01T00:00:00+00:00'})
                members.append({'model':model,'seed':'deterministic' if model in engine.NAIVES else 670101,
                    'fit_directory':model,'fit_origin':'2023-03-18','fit_outcome_max':'2023-03-10',
                    'configuration_selection_outcome_max':'2023-03-10','run_id':model,
                    'checkpoint_sha256':'a'*64,'scaler_sha256':'b'*64,'config_sha256':'c'*64,
                    'checkpoint_closed_at_utc':'2026-01-02T00:00:00+00:00'})
            context={'output':output,'root':ROOT,'inputs':{}}
            with patch.object(engine,'development',return_value=data), \
                 patch.object(engine,'verify_job',return_value={'artifacts':{'training_evaluation.npz':'fixture'}}):
                rows=engine.training_rows(context,{'members':members},source_stage='synthetic_fixture')
            masks=[[r['eligible'] for r in rows if r['model']==model] for model in ['GRU-E2','Persistence','SeasonalNaive']]
            self.assertEqual(masks[0],masks[1]);self.assertEqual(masks[0],masks[2])
            self.assertEqual(next(i for i,v in enumerate(masks[0]) if v),14)
            self.assertFalse(masks[0][36]);self.assertFalse(masks[0][42])
            self.assertTrue(all(r['y_pred'] is None and r['y_true'] is None for r in rows if not r['eligible']))


if __name__=='__main__':unittest.main()
