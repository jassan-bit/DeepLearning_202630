"""Read-only registration gates; no model construction, registration or fitting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_filipe_confirmatory_d7 as study


class ConfirmatoryRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = study.make_plan()

    def setUp(self):
        self.plan = deepcopy(self.template)
        for owner, name in [(study.fitting,'optimization_fit'),(study.fitting,'build_model'),
                            (study.fitting,'infer'),(study,'write'),(study.core,'run_trajectory')]:
            guard=patch.object(owner,name,side_effect=AssertionError('Validation called '+name))
            guard.start(); self.addCleanup(guard.stop)

    def test_actual_twelve_folds_are_past_only_and_cover_exactly_2024(self):
        manifests=study.validate_inputs(self.plan)
        self.assertEqual(set(manifests),{7})
        self.assertEqual(len(manifests[7]['folds']),12)
        outcomes=[]
        for fold in manifests[7]['folds']:
            outcomes.extend(fold['test_indices'])
            self.assertLessEqual(max(fold['train_indices']),fold['fit_origin_index']-8)
            self.assertLessEqual(max(fold['inner_train_indices']),min(fold['inner_validation_indices'])-8)
            self.assertEqual(len(fold['inner_validation_indices']),83)
        self.assertEqual(outcomes,list(range(365,731)))
        self.assertEqual(len(set(outcomes)),366)

    def test_actual_runner_plan_is_compatible_with_registered_analyzer(self):
        import analyze_filipe_confirmatory_d7 as analyzer
        analyzer.validate_plan(self.plan)
        self.assertEqual(self.plan['analysis'],study.ANALYSIS)
        self.assertEqual(self.plan['maximum_definitive_seeds'],300)

    def test_fresh_namespaces_include_month_refit_and_minibatch_roles(self):
        seeds=self.plan['variance_extension_seeds']
        self.assertEqual(seeds,list(range(6301,6341,2)))
        old=study.seed_namespace(study.previous_seeds())
        self.assertTrue(study.seed_namespace(seeds).isdisjoint(old))
        final_seeds=[50000001+2*i for i in range(10000)]
        final=study.seed_namespace(final_seeds)
        self.assertEqual(len(final),10000*12*4)
        self.assertEqual(len(study.seed_namespace(seeds)),20*12*4)
        self.assertTrue(final.isdisjoint(old|study.seed_namespace(seeds)))
        self.assertEqual(len(study.jobs_for(self.plan,seeds)),60)
        for model in self.plan['models']:
            self.assertEqual([j['seed'] for j in study.jobs_for(self.plan,seeds) if j['model']==model],seeds)

    def test_previous_seed_including_cross_role_reuse_is_rejected(self):
        with patch.object(study,'previous_seeds',return_value={6301}):
            with self.assertRaisesRegex(ValueError,'must be fresh'):
                study.validate_plan(self.plan)
        # A previous minibatch state can collide with a proposed initialization.
        with patch.object(study,'previous_seeds',return_value={6284}):
            with self.assertRaisesRegex(ValueError,'must be fresh'):
                study.validate_plan(self.plan)

    def test_fixed_model_recipe_target_and_boundaries(self):
        changes=[lambda p:p['models']['MLP'].update(width=32),
                 lambda p:p['training'].update(max_epochs=101),
                 lambda p:p.update(histories=[7,14]),
                 lambda p:p.update(variant='absolute_log'),
                 lambda p:p.update(target_return_days=14),
                 lambda p:p.update(forecast_horizon_days=2),
                 lambda p:p.update(workers=3),
                 lambda p:p.update(test_files_opened_allowed=1),
                 lambda p:p.update(raw_archives_opened_allowed=1),
                 lambda p:p.update(variance_extension_seeds=list(range(6201,6221))),
                 lambda p:p.update(definitive_seed_start=6301),
                 lambda p:p.update(definitive_seed_step=1),
                 lambda p:p.update(original_pilot_seeds=list(range(6101,6106))),
                 lambda p:p['analysis'].update(bootstrap_replicates=99)]
        for change in changes:
            changed=deepcopy(self.plan);change(changed)
            with self.subTest(change=change),self.assertRaises(ValueError):
                study.validate_plan(changed)

    def test_dimensioning_uses_twenty_new_seeds_and_exact_registered_family(self):
        changes=[lambda p:p['primary_pairs'].pop(),
                 lambda p:p['dimensioning'].update(variance_extension_n=25),
                 lambda p:p['dimensioning'].update(sd_source='pooled25'),
                 lambda p:p['dimensioning'].update(primary_family_size=12),
                 lambda p:p['dimensioning'].update(alpha_global=.1),
                 lambda p:p['dimensioning'].update(alpha_each=.05),
                 lambda p:p['dimensioning'].update(power_target=.9),
                 lambda p:p['dimensioning'].update(execution_floor=5),
                 lambda p:p['dimensioning'].update(sd_upper_confidence=.9),
                 lambda p:p['dimensioning'].update(maximum_planning_R=300),
                 lambda p:p['dimensioning'].update(delta_absolute=.1)]
        for change in changes:
            changed=deepcopy(self.plan);change(changed)
            with self.subTest(change=change),self.assertRaises(ValueError):
                study.validate_plan(changed)

    def test_changed_source_and_code_hashes_fail_before_development_loading(self):
        for category in ('source_sha256','code_sha256'):
            changed=deepcopy(self.plan);changed[category][next(iter(changed[category]))]='0'*64
            with self.subTest(category=category),patch.object(study.pilot,'load_development',side_effect=AssertionError('Too late source check')):
                with self.assertRaisesRegex(ValueError,'Registered source changed'):
                    study.validate_inputs(changed)

    def test_missing_source_guard_is_rejected(self):
        self.plan['source_sha256'].pop(study.DECISION)
        with self.assertRaisesRegex(ValueError,'source guards'):
            study.validate_plan(self.plan)

    def test_actual_fold_with_broken_embargo_is_rejected(self):
        original=study.read
        def changed_read(path):
            value=original(path)
            if Path(path)==study.ROOT/study.FOLD_SOURCE:
                value=deepcopy(value)
                value['folds'][0]['train_indices'][-1]=value['folds'][0]['fit_origin_index']-1
            return value
        with patch.object(study,'read',side_effect=changed_read):
            with self.assertRaisesRegex(ValueError,'Invalid past-only fold'):
                study.validate_inputs(self.plan)

    def test_lock_requires_complete_extension_and_matching_analysis_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); phase=out/'variance_extension';phase.mkdir()
            (phase/'runs.csv').write_text('model,seed,rmse\nMLP,6301,.1\n')
            (phase/'execution_summary.json').write_text(json.dumps({'completed_at_utc':'2026-11-05T01:00:00+00:00'}))
            lock={'common_R':30,'registered_protocol_sha256':'a'*64,
                  'analysis_code_sha256':self.plan['code_sha256']['scripts/analyze_filipe_confirmatory_d7.py'],
                  'locked_at_utc':'2026-11-05T02:00:00+00:00',
                  'variance_extension_runs_sha256':study.digest(phase/'runs.csv'),
                  'variance_extension_summary_sha256':study.digest(phase/'execution_summary.json')}
            planpath=out/'protocol.json';planpath.write_text('{}')
            # Temporary files need a temporary ROOT to preserve safe relative guards.
            original_root=study.ROOT
            (out/study.PILOT_RECORDS[-1]).parent.mkdir(parents=True,exist_ok=True)
            (out/study.PILOT_RECORDS[-1]).write_bytes((original_root/study.PILOT_RECORDS[-1]).read_bytes())
            lock['input_sha256']={str(p.relative_to(out)):study.digest(p) for p in [planpath,out/study.PILOT_RECORDS[-1],phase/'runs.csv',phase/'execution_summary.json']}
            path=out/'locked_design.json';path.write_text(json.dumps(lock));context={'protocol_sha256':'a'*64}
            with patch.object(study,'OUT',out),patch.object(study,'ROOT',out),patch.object(study,'PROTOCOL',planpath),patch.object(study,'LOCK',path),patch('analyze_filipe_confirmatory_d7.verify_dimension_lock',return_value={}):
                self.assertEqual(study.validate_lock(self.plan,context)['common_R'],30)
                for key,bad in [('common_R',None),('common_R',29),('common_R',10001),('common_R',True),
                                ('registered_protocol_sha256','b'*64),('analysis_code_sha256','b'*64),
                                ('variance_extension_runs_sha256','b'*64),
                                ('variance_extension_summary_sha256','b'*64),
                                ('locked_at_utc','2026-11-05T00:00:00+00:00')]:
                    changed={**lock,key:bad};path.write_text(json.dumps(changed))
                    with self.subTest(key=key,value=bad),self.assertRaises(ValueError):
                        study.validate_lock(self.plan,context)


if __name__=='__main__': unittest.main()
