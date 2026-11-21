"""Corruption and false-claim guards using tiny synthetic metadata fixtures only."""
from pathlib import Path
import hashlib
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import verify_package as v


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def meta(path, rel):
    b = path.read_bytes()
    return {'path':rel, 'bytes':len(b), 'sha256':hashlib.sha256(b).hexdigest()}


def finalize(root):
    components = []
    for name in ['calibration', 'retrospective', 'guide_power', 'bundle_tools']:
        d = root/name
        records = [meta(p,p.relative_to(d).as_posix()) for p in sorted(d.rglob('*'))
                   if p.is_file() and p != d/'MANIFEST.json']
        write(d/'MANIFEST.json', {'schema_version':1, 'files':records})
        components.append({'manifest':name+'/MANIFEST.json', 'sha256':v.digest(d/'MANIFEST.json')})
    records = [meta(p,p.relative_to(root).as_posix()) for p in sorted(root.rglob('*'))
               if p.is_file() and p != root/'PACKET_MANIFEST.json']
    write(root/'PACKET_MANIFEST.json', {'schema_version':1, 'source_base_commit':v.BASE,
          'files':records, 'components':components, 'new_forecasting_model_fits':0,
          'new_forecasting_model_forward_passes':0, 'new_market_predictions':0,
          'R3_accredited':False, 'confirmation_enabled':False, 'human_execution_claimed':False})


def fixture(root, partial=False):
    write(root/'initial_state.json', {'base_commit':v.BASE,
          'preserve':dict(additive_jobs=796,V3_fits=30,transfer_trajectories=51,
                          transfer_rows=20400,V4_fits=20,V4_rows=1210,V3_NO_GO=True,V4_abandoned=True),
          'oral_score_provided_by_author':4.5, 'observed_defense_claimed':False,
          'new_model_fits_allowed':False,'new_forecasts_allowed':False,'R3_accredited':False})
    cases = [dict(id='case'+str(i),replicates=n) for i,n in enumerate([256,128,128,128])]
    write(root/'calibration/VALIDATION_PLAN.json', {'planned_replicates':640,'cases':cases,
          'settings':{'B':999},'validation_seed':91,'old_validation_seed_not_reused':7,
          'confirmation_allowed':False,'R3_accredited':False,'new_market_fits':0,'new_market_predictions':0})
    (root/'calibration/source.py').write_text('# inert fixture; no statistical implementation\n')
    write(root/'calibration/CODE_MANIFEST.json', {'files':[meta(root/'calibration/source.py','source.py')]})
    anchor = {'status':'MCS_bootstrap_t_code_and_plan_published_before_validation', 'commit':'a'*40,
          'observed_by_root_before_validation':True,
          'plan_SHA256':v.digest(root/'calibration/VALIDATION_PLAN.json'),
          'source_manifest_SHA256':v.digest(root/'calibration/CODE_MANIFEST.json')}
    write(root/'calibration/PUBLIC_ANCHOR.json',anchor)
    status='resource_stop' if partial else 'completed'; n=256 if partial else 640
    common={'source_manifest_SHA256':anchor['source_manifest_SHA256'],'public_anchor_commit':anchor['commit']}
    write(root/'calibration/validation_fresh/execution_receipt.json', {**common,'plan_SHA256':anchor['plan_SHA256'],
          'source_hashes_still_match':True,'completed_replicates':n,'planned_replicates':640,
          'actual_crossed_bootstrap_draws':n*999,'old_control_extra_bootstrap_draws':0,'status':status,
          'errors':[],'new_market_fits':0,'new_market_predictions':0,'R3_accredited':False,
          'human_execution_claimed':False,'source_implementation_changed_after_results':False})
    cc=cases[:1] if partial else cases
    write(root/'calibration/validation_fresh/validation_summary.json', {**common,'status':status,
          'complete':not partial,'cases':[{'id':c['id'],'planned_replicates':c['replicates'],
                                        'completed_replicates':c['replicates']} for c in cc],
          'three_primary_pairwise_tests_accredited':False,'complete_three_way_power_computed':False,
          'final_market_calibration_certified':False,'final_market_power_certified':False,
          'R3_accredited':False,'bootstrap_variants_are_not_three_distinct_primary_tests':True})
    summary_path=root/'calibration/validation_fresh/validation_summary.json'
    summary=json.loads(summary_path.read_text());summary.update(new_market_fits=0,new_market_predictions=0)
    write(summary_path,summary)
    with gzip.open(root/'calibration/validation_fresh/per_replicate.jsonl.gz','wt',encoding='utf-8') as h:
        for c in cc:
            for i in range(c['replicates']):
                h.write(json.dumps({'case':c['id'],'replicate':i,'stage':'fresh_controlled_validation',
                    'q_SHA256':'b'*64,'result':{'shape':[12,128,20],'pairwise_methods':2,
                    'pairwise':[{}]*19,'third_primary_method_accredited':False,
                    'confirmed_valid_for_market':False,'R3_accredited':False,
                    'new_market_fits':0,'new_market_predictions':0}})+'\n')
    write(root/'retrospective/results/execution_receipt.json', {
          'status':'completed_retrospective_adequacy_not_confirmation','new_forecasting_model_fits':0,
          'new_forecasting_model_forward_passes':0,'new_market_predictions':0,'new_bootstrap_draws':0,
          'R3_accredited':False,'confirmation_enabled':False,'human_execution_claimed':False,
          'primary_cross_seed_R10_guard_relaxed':False,'frozen_inputs_after_execution_unchanged':True})
    write(root/'retrospective/results/analysis_summary.json', {
          'trajectories':22,'recipes':6,'prediction_points':1210,'diagnostic_family_slots':88,
          'pointwise_asymptotic_mean_error_intervals':44,'new_auxiliary_OLS_regressions':44,
          'primary_minimum_seeds_required':10,'neural_seed_count_observed':5,
          'computed_diagnostic_p_values':88,'unsupported_reserved_slots':0,
          'primary_cross_seed_inference_run':False,'formal_overfit_adequacy_pass':False,
          'V4_predictive_interval_coverage_or_PIT_available':False,'V4_calibrator_invented':False,
          'simultaneous_coverage_claimed':False,'R3_accredited':False,'V3_original_NO_GO_preserved':True,
          'V4_original_abandon_decision_preserved':True,'new_forecaster_fits':0,
          'new_forecaster_forward_passes':0,'new_bootstrap_draws':0})
    write(root/'guide_power/EXECUTED_POWER_INPUT_AUDIT.json', {
          'saved_scenarios_reconciled':486,'older_distinct_planning_sources_reverified':71,
          'old_V3_available_development_variances':3,'old_V3_missing_development_variances':16,
          'old_V3_future_cells_missing':76,'V4_future_cells_missing':84,
          'new_V4_D00_descriptive_pairs':5,'new_V4_descriptive_covariance_entries':25,
          'admitted_future_panels':0,'final_R':None,'final_T':None,'new_V4_refs_are_future_inputs':False,
          'planning340fits_authorized':False,'full_battery_power_established':False,
          'guide_joint90_required':False,'zero_imputation_used':False,'confirmation_enabled':False,'R3_accredited':False})
    power_path=root/'guide_power/EXECUTED_POWER_INPUT_AUDIT.json';power=json.loads(power_path.read_text())
    power.update(new_model_fits=0,new_model_predictions=0,new_statistical_resamples=0,new_market_p_values=0)
    write(power_path,power)
    write(root/'bundle_tools/unit-fixture.json', {'scope':'synthetic metadata only, no experiments'})
    finalize(root)


class PackageGuards(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)/'packet'
        self.root.mkdir(); fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def change(self, path, **changes):
        p=self.root/path; data=json.loads(p.read_text());data.update(changes);write(p,data);finalize(self.root)
    def fails(self):
        with self.assertRaises(v.VerificationError):v.verify_package(self.root)
    def test_complete_metadata_passes_without_execution(self):
        r=v.verify_package(self.root);self.assertEqual(r['status'],'PASS')
        self.assertFalse(r['analysis_or_sampling_executed_by_verifier'])
    def test_honest_resource_stop_is_not_mislabeled_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);fixture(p,True);r=v.verify_package(p)
            self.assertEqual((r['controlled_validation_status'],r['new_validation_replicates']),('resource_stop',256))
    def test_corrupt_payload(self):
        (self.root/'calibration/source.py').write_text('changed');self.fails()
    def test_missing_payload(self):
        (self.root/'calibration/source.py').unlink();self.fails()
    def test_unlisted_member(self):
        (self.root/'unexpected.txt').write_text('extra');self.fails()
    def test_forbidden_raw_weight_even_with_refreshed_manifests(self):
        (self.root/'calibration/checkpoint.pt').write_bytes(b'not real weights');finalize(self.root);self.fails()
    def test_symlink(self):
        (self.root/'alias').symlink_to(self.root/'calibration/source.py');self.fails()
    def test_directory_symlink_rejected_before_hashing_outside(self):
        outside=Path(self.temp.name)/'outside';outside.mkdir();(outside/'data').write_text('outside')
        (self.root/'linked').symlink_to(outside,target_is_directory=True)
        p=self.root/'PACKET_MANIFEST.json';d=json.loads(p.read_text())
        d['files'].append(meta(outside/'data','linked/data'));write(p,d)
        original=v.digest
        def inside(path):
            self.assertTrue(path.resolve().is_relative_to(self.root.resolve()))
            return original(path)
        with mock.patch.object(v,'digest',side_effect=inside):self.fails()
    def test_traversal_in_packet_manifest(self):
        p=self.root/'PACKET_MANIFEST.json';d=json.loads(p.read_text());d['files'][0]['path']='../escape';write(p,d);self.fails()
    def test_partial_promoted_complete(self):
        self.change('calibration/validation_fresh/execution_receipt.json',completed_replicates=256,
                    actual_crossed_bootstrap_draws=256*999);self.fails()
    def test_R3_claim_rejected_even_with_rehashed_bytes(self):
        self.change('calibration/validation_fresh/validation_summary.json',R3_accredited=True);self.fails()
    def test_validation_summary_fit_counter_checked(self):
        self.change('calibration/validation_fresh/validation_summary.json',new_market_fits=1);self.fails()
    def test_retrospective_receipt_fit_counter_checked(self):
        self.change('retrospective/results/execution_receipt.json',new_forecasting_model_fits=1);self.fails()
    def test_retrospective_receipt_R3_claim_checked(self):
        self.change('retrospective/results/execution_receipt.json',R3_accredited=True);self.fails()
    def test_power_counter_checked(self):
        self.change('guide_power/EXECUTED_POWER_INPUT_AUDIT.json',new_model_fits=1);self.fails()
    def test_resource_stop_cannot_hide_execution_error(self):
        self.change('calibration/validation_fresh/execution_receipt.json',status='resource_stop',errors=[{'type':'ValueError'}]);self.fails()
    def test_variants_not_three_primary_methods(self):
        self.change('calibration/validation_fresh/validation_summary.json',three_primary_pairwise_tests_accredited=True);self.fails()
    def test_unsupported_diagnostic_slots_not_dropped(self):
        self.change('retrospective/results/analysis_summary.json',computed_diagnostic_p_values=87);self.fails()
    def test_R5_not_used_as_R10_primary_inference(self):
        self.change('retrospective/results/analysis_summary.json',primary_cross_seed_inference_run=True);self.fails()
    def test_no_invented_interval_calibrator(self):
        self.change('retrospective/results/analysis_summary.json',V4_calibrator_invented=True);self.fails()
    def test_no_invented_prospective_variance(self):
        self.change('guide_power/EXECUTED_POWER_INPUT_AUDIT.json',new_V4_refs_are_future_inputs=True);self.fails()
    def test_no_V4_planning_restart(self):
        self.change('guide_power/EXECUTED_POWER_INPUT_AUDIT.json',planning340fits_authorized=True);self.fails()
    def test_no_final_R_invented(self):
        self.change('guide_power/EXECUTED_POWER_INPUT_AUDIT.json',final_R=10);self.fails()
    def test_no_observed_oral_claim(self):
        self.change('initial_state.json',observed_defense_claimed=True);self.fails()
    def test_initial_policy_cannot_authorize_forecast_training(self):
        self.change('initial_state.json',new_model_fits_allowed=True);self.fails()
    def test_plan_cannot_authorize_confirmation(self):
        self.change('calibration/VALIDATION_PLAN.json',confirmation_allowed=True);self.fails()
    def test_anchor_hash_changed(self):
        self.change('calibration/PUBLIC_ANCHOR.json',plan_SHA256='0'*64);self.fails()
    def test_no_validation_RNG_reuse(self):
        self.change('calibration/VALIDATION_PLAN.json',validation_seed=7);self.fails()
    def test_source_changed_after_results(self):
        self.change('calibration/validation_fresh/execution_receipt.json',source_hashes_still_match=False);self.fails()
    def test_missing_controlled_raw_replication(self):
        p=self.root/'calibration/validation_fresh/per_replicate.jsonl.gz'
        with gzip.open(p,'rt',encoding='utf-8') as h:lines=h.readlines()
        with gzip.open(p,'wt',encoding='utf-8') as h:h.writelines(lines[:-1])
        finalize(self.root);self.fails()
    def test_duplicate_controlled_replication(self):
        p=self.root/'calibration/validation_fresh/per_replicate.jsonl.gz'
        with gzip.open(p,'rt',encoding='utf-8') as h:lines=h.readlines()
        lines[-1]=lines[-2]
        with gzip.open(p,'wt',encoding='utf-8') as h:h.writelines(lines)
        finalize(self.root);self.fails()
    def test_controlled_replicate_claim_not_only_summary_checked(self):
        p=self.root/'calibration/validation_fresh/per_replicate.jsonl.gz'
        with gzip.open(p,'rt',encoding='utf-8') as h:lines=h.readlines()
        item=json.loads(lines[0]);item['result']['R3_accredited']=True;lines[0]=json.dumps(item)+'\n'
        with gzip.open(p,'wt',encoding='utf-8') as h:h.writelines(lines)
        finalize(self.root);self.fails()
    def test_duplicate_JSON_key(self):
        p=self.root/'initial_state.json';p.write_text('{"base_commit":"x","base_commit":"y"}');finalize(self.root);self.fails()
    def test_JSON_root_array_rejected_as_corruption(self):
        write(self.root/'initial_state.json',[]);finalize(self.root);self.fails()
    def test_cli_report_inside_packet_never_written(self):
        target=self.root/'forbidden-report.json';before={p.relative_to(self.root) for p in self.root.rglob('*')}
        r=subprocess.run([sys.executable,str(Path(v.__file__)),'--packet',str(self.root),'--output',str(target)],capture_output=True,text=True)
        self.assertEqual(r.returncode,1);self.assertFalse(target.exists())
        self.assertEqual(before,{p.relative_to(self.root) for p in self.root.rglob('*')})


if __name__=='__main__':unittest.main()
