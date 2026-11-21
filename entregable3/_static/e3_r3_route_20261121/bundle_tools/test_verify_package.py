"""Corruption and claim-boundary guards; no model/statistical/network execution."""
import hashlib
import gzip
from datetime import datetime,timedelta
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import verify_package as v


def component_source(component):
    here=Path(__file__).resolve()
    return next(p for p in [here.parent.parent,here.parent.parent.parent] if (p/component).is_dir())


class ClosureConsistencyGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.packet={'source_base_commit':v.BASE,'new_forecasting_model_fits':0,
                     'new_forecasting_model_forward_passes':0,'new_market_predictions':0,
                     'R3_accredited':False,'confirmation_enabled':False,'human_execution_claimed':False,
                     'AI_assistance_declared':True}
        self.counts={'new_controlled_replicates':17,'new_completed_replicate_bootstrap_draws':17*999}
        self.closure={**self.packet,**self.counts}
    def tearDown(self):self.tmp.cleanup()
    def verify(self):
        (self.root/'CLOSURE_STATE.json').write_text(json.dumps(self.closure))
        return v.check_closure(self.root,self.packet,self.counts)
    def test_summary_counts_follow_verified_input_not_a_fixed_desired_total(self):self.verify()
    def test_summary_cannot_accredit_R3(self):
        self.closure['R3_accredited']=True
        with self.assertRaises(v.VerificationError):self.verify()
    def test_summary_cannot_report_more_replicates_than_verified(self):
        self.closure['new_controlled_replicates']=18
        with self.assertRaises(v.VerificationError):self.verify()


def rate(events):
    # Deliberately broad fixture bounds, not a statistical interval calculation.
    return {'events':events,'replicates':1,'rate':float(events),'CI95':[0.,1.],
            'simultaneous_interval':False}


def fwer(events,nominal=.05):
    return {**rate(events),'nominal':nominal,'exploratory_tolerance':.03,'screen':'inconclusive',
            'exact_nominal_level_certified':False,'market_calibration_certified':False}


def coverage(events):
    return {**rate(events),'nominal':.95,'exploratory_tolerance':.03,'screen':'inconclusive',
            'simultaneous_coverage_claimed':False}


class SavedDecisionRecountGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.output=self.root/'statistical_revision/validation_new';self.output.mkdir(parents=True)
        bases=[1.]*10+[1.23]*10
        self.case={'id':'fixture_true_partial_equality','replicates':1,'R':12,'T':128,'M':20,'bases':bases}
        truth=[x-1 for x in bases[1:]];names=[f'rival_{i:02d}' for i in range(1,20)]
        pairwise=[]
        for j,name in enumerate(names):
            pairwise.append({'rival':name,'P1_Holm':.01 if j==0 else 1.,
                             'P2_Holm':.01 if j==0 else 1.,
                             'P1_basic_frozen_control_Holm38':1.,'P2_frozen_control_Holm38':1.,
                             'P1_supported':True,'P2_supported':True,'P1_CI95':[-1.,1.],
                             'P2_CI95':[-1.,1.]})
        self.raw={'case':self.case['id'],'replicate':0,'bases':bases,'analytic_longrun_theta':truth,
                  'result':{'shape':[12,128,20],'bootstrap_studentizers_recomputed':999,
                            'pairwise_methods':2,'P1_CI_absolute_pivot_order_rank':951,
                            'new_market_fits':0,'new_market_predictions':0,
                            'third_primary_method_accredited':False,'confirmed_valid_for_market':False,
                            'R3_accredited':False,'pairwise':pairwise,
                            'MCS_bootstrap_t_frozen_control':{'retained':['candidate']+names,
                                                             'initial_global_p':1.}}}
        self.item={'id':self.case['id'],'completed_replicates':1,'complete':True,
                   'R3_accredited':False,'final_market_power_certified':False,
                   'true_null_rivals':names[:9],
                   'known_true_null_component_FWER_Holm38':fwer(1),
                   'old_basic_control_known_true_null_FWER_Holm38':fwer(0),
                   'known_true_null_two_method_intersection_event':rate(1),
                   'known_true_null_FWER_by_component_within_same_Holm38':{'P1':rate(1),'P2':rate(1)},
                   'paired_old_basic_mechanism_diagnostic':{'old_only_FWER':0,'new_only_FWER':1,
                                                            'causal_explanation_or_superiority_certified':False},
                   'per_rival':[{'rival':name,'methods':{m:{'pointwise_CI95_coverage':
                                                           coverage(1)}
                                                        for m in ['P1','P2']}} for name in names],
                   'MCS_bootstrap_t_frozen_control':{
                       'all_true_best_retained':rate(1),'any_true_best_excluded':rate(0),
                       'candidate_retained':rate(1),'unique_candidate_set':rate(0),
                       'false_unique_candidate':rate(0),'third_pairwise_p_provided':False,
                       'initial_global_rejection':rate(0),'unsupported_replicates':0,
                       'true_best_models':['candidate']+names[:9]},
                   'complete_three_method_power_computed':False,'simultaneous_CI95_claimed':False}
        self.receipt={'status':'resource_stop','completed_replicates':1,'attempted_replicates':1,'errors':[]}
    def tearDown(self):self.tmp.cleanup()
    def recount(self):
        with gzip.open(self.output/'per_replicate.jsonl.gz','wt') as stream:stream.write(json.dumps(self.raw)+'\n')
        (self.output/(self.case['id']+'_summary.json')).write_text(json.dumps(self.item))
        return v.verify_raw_validation(self.root,{'cases':[self.case]},self.receipt,{'cases':[self.item]})
    def test_true_partial_null_fixture_passes_without_any_sampling(self):self.recount()
    def test_false_nulls_cannot_be_relabelled_true(self):
        self.item['true_null_rivals']=[f'rival_{i:02d}' for i in range(1,20)]
        with self.assertRaises(v.VerificationError):self.recount()
    def test_true_null_false_rejection_cannot_be_erased(self):
        self.item['known_true_null_component_FWER_Holm38']=fwer(0)
        with self.assertRaises(v.VerificationError):self.recount()
    def test_raw_completed_count_must_match_receipt(self):
        self.receipt['completed_replicates']=0
        with self.assertRaises(v.VerificationError):self.recount()
    def test_nonfinite_raw_adjusted_p_is_rejected(self):
        self.raw['result']['pairwise'][0]['P1_Holm']=float('inf')
        with self.assertRaises(v.VerificationError):self.recount()
    def test_unsupported_slot_cannot_become_an_interval(self):
        self.raw['result']['pairwise'][0]['P1_supported']=False
        with self.assertRaises(v.VerificationError):self.recount()
    def test_MCS_best_set_coverage_cannot_be_falsely_reported(self):
        self.item['MCS_bootstrap_t_frozen_control']['all_true_best_retained']=rate(0)
        with self.assertRaises(v.VerificationError):self.recount()
    def test_partial_replicates_cannot_be_reported_complete(self):
        self.receipt['status']='completed'
        with self.assertRaises(v.VerificationError):self.recount()
    def test_pointwise_coverage_cannot_be_promoted_to_simultaneous(self):
        self.item['per_rival'][0]['methods']['P1']['pointwise_CI95_coverage']['simultaneous_coverage_claimed']=True
        with self.assertRaises(v.VerificationError):self.recount()
    def test_FWER_screen_cannot_pass_when_saved_bounds_show_inflation(self):
        screen=self.item['known_true_null_component_FWER_Holm38']
        screen.update(CI95=[.06,1.],screen='exploratory_tolerance_screen_passed')
        with self.assertRaises(v.VerificationError):self.recount()
    def test_undercoverage_bounds_cannot_be_relabelled_as_pass(self):
        self.raw['result']['pairwise'][0]['P1_CI95']=[2.,3.]
        record=coverage(0);record.update(CI95=[0.,.94],screen='exploratory_tolerance_screen_passed')
        self.item['per_rival'][0]['methods']['P1']['pointwise_CI95_coverage']=record
        with self.assertRaises(v.VerificationError):self.recount()
    def test_MCS_initial_global_rejections_must_match_raw_p(self):
        self.raw['result']['MCS_bootstrap_t_frozen_control']['initial_global_p']=.01
        with self.assertRaises(v.VerificationError):self.recount()
    def test_MCS_global_null_screen_must_match_saved_bounds(self):
        self.case['bases']=[1.]*20;self.raw['bases']=[1.]*20;self.raw['analytic_longrun_theta']=[0.]*19
        names=[f'rival_{i:02d}' for i in range(1,20)];self.item['true_null_rivals']=names
        self.raw['result']['MCS_bootstrap_t_frozen_control']['initial_global_p']=.01
        mcs=self.item['MCS_bootstrap_t_frozen_control'];mcs['true_best_models']=['candidate']+names
        record=fwer(1,.10);record.update(CI95=[.11,1.],screen='exploratory_tolerance_screen_passed')
        mcs['initial_global_rejection']=record
        with self.assertRaises(v.VerificationError):self.recount()


class StoppedExecutionMetadataGuards(unittest.TestCase):
    """Invented zero-result transport fixtures; never reads unfinished results."""
    def setUp(self):
        source=component_source('statistical_revision')/'statistical_revision'
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.component=self.root/'statistical_revision';self.component.mkdir()
        code=json.loads((source/'CODE_MANIFEST.json').read_text())
        for name in [x['path'] for x in code['files']]+['CODE_MANIFEST.json','PUBLIC_ANCHOR.json']:
            shutil.copyfile(source/name,self.component/name)
        self.anchor=json.loads((self.component/'PUBLIC_ANCHOR.json').read_text())
        start=datetime.fromisoformat(self.anchor['observed_at_utc'])+timedelta(seconds=1)
        self.output=self.component/'validation_new';self.output.mkdir()
        self.receipt={'status':'resource_stop','completed_replicates':0,'attempted_replicates':0,
                      'errors':[],'planned_replicates':1664,'planned_crossed_bootstrap_draws':1662336,
                      'plan_SHA256':v.digest(self.component/'VALIDATION_PLAN.json'),
                      'source_manifest_SHA256':v.digest(self.component/'CODE_MANIFEST.json'),
                      'public_anchor_commit':self.anchor['commit'],'started_at_utc':start.isoformat(),
                      'finished_at_utc':(start+timedelta(seconds=600)).isoformat(),
                      'actual_crossed_bootstrap_draws':0,'completed_replicate_bootstrap_draws':0,
                      'failed_attempt_bootstrap_draws_exact':0,'failed_attempt_bootstrap_draws_upper_bound':0,
                      'control_extra_bootstrap_draws':0,'source_hashes_still_match':True,
                      'source_implementation_changed_after_results':False,'R3_accredited':False,
                      'new_market_fits':0,'new_market_forwards':0,'new_market_predictions':0,
                      'runtime':{'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}}
        self.summary={'status':'resource_stop','complete':False,'cases':[],
                      'public_anchor_commit':self.anchor['commit'],
                      'source_manifest_SHA256':self.receipt['source_manifest_SHA256'],
                      'three_primary_pairwise_tests_accredited':False,'R3_accredited':False,
                      'new_market_fits':0,'new_market_forwards':0,'new_market_predictions':0}
        self.decision={'status':'incomplete_no_family_approval','complete':False,
                       'known_true_null_FWER_screens':[],'primary_market_admissibility_certified':False,
                       'exact_nominal_level_certified':False,'third_primary_method_accredited':False,
                       'complete_three_method_power_computed':False,'confirmed_valid_for_market':False,
                       'confirmation_allowed':False,'R3_accredited':False,'new_market_fits':0,
                       'new_market_forwards':0,'new_market_predictions':0,
                       'V3_NO_GO_unchanged':True,'V4_abandonment_unchanged':True}
    def tearDown(self):self.tmp.cleanup()
    def verify(self):
        (self.component/'PUBLIC_ANCHOR.json').write_text(json.dumps(self.anchor))
        with gzip.open(self.output/'per_replicate.jsonl.gz','wt') as stream:stream.write('')
        self.receipt['raw_log_SHA256']=v.digest(self.output/'per_replicate.jsonl.gz')
        for name,value in [('execution_receipt.json',self.receipt),('validation_summary.json',self.summary),
                           ('RESULTS_DECISION.json',self.decision)]:
            (self.output/name).write_text(json.dumps(value))
        return v.check_statistical_revision(self.root)
    def test_honest_stopped_zero_result_fixture_is_partial(self):
        result=self.verify();self.assertEqual(result['new_controlled_replicates'],0)
        self.assertEqual(result['controlled_validation_status'],'resource_stop')
    def test_partial_execution_cannot_be_promoted_complete(self):
        self.receipt['status']='completed';self.summary.update(status='completed',complete=True)
        self.decision['complete']=True
        with self.assertRaises(v.VerificationError):self.verify()
    def test_fabricated_zero_anchor_is_rejected(self):
        self.anchor['commit']='0'*40
        with self.assertRaises(v.VerificationError):self.verify()
    def test_wrong_anchor_plan_hash_is_rejected(self):
        self.anchor['plan_SHA256']='0'*64
        with self.assertRaises(v.VerificationError):self.verify()
    def test_late_anchor_cannot_be_a_preregistered_source(self):
        self.anchor['observed_at_utc']=self.receipt['finished_at_utc']
        with self.assertRaises(v.VerificationError):self.verify()
    def test_statistical_study_cannot_authorize_confirmation(self):
        self.decision['confirmation_allowed']=True
        with self.assertRaises(v.VerificationError):self.verify()
    def test_market_prediction_counter_must_remain_zero(self):
        self.receipt['new_market_predictions']=1
        with self.assertRaises(v.VerificationError):self.verify()


class ByteInventoryGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'payload.txt').write_bytes(b'public fixture\n')
        self.entry={'path':'payload.txt','bytes':15,
                    'sha256':hashlib.sha256(b'public fixture\n').hexdigest()}
    def tearDown(self):self.tmp.cleanup()
    def test_valid_bytes_are_accepted(self):
        names,total=v.check_entries(self.root,[self.entry]);self.assertEqual(names,{'payload.txt'})
        self.assertEqual(total,15)
    def test_modified_payload_is_rejected(self):
        (self.root/'payload.txt').write_bytes(b'forged receipt\n')
        with self.assertRaises(v.VerificationError):v.check_entries(self.root,[self.entry])
    def test_missing_payload_is_rejected(self):
        (self.root/'payload.txt').unlink()
        with self.assertRaises(v.VerificationError):v.check_entries(self.root,[self.entry])
    def test_duplicate_manifest_member_is_rejected(self):
        with self.assertRaises(v.VerificationError):v.check_entries(self.root,[self.entry,self.entry])
    def test_symlink_payload_is_rejected(self):
        (self.root/'alias.txt').symlink_to(self.root/'payload.txt')
        with self.assertRaises(v.VerificationError):v.actual_members(self.root)
    def test_unsafe_or_private_paths_are_rejected(self):
        for path in ['.','../payload.txt','/tmp/payload.txt','x/../payload.txt','x\\payload.txt',
                     'guide_candidate/primary/guide_full_extraction.txt','weights.pt','book.whl',
                     '__pycache__/x.pyc','guide.pdf','x//payload.txt']:
            with self.subTest(path=path),self.assertRaises(v.VerificationError):v.safe_path(path)
    def test_duplicate_json_keys_are_rejected_in_nested_receipt(self):
        with self.assertRaises(v.VerificationError):v.decode_json('{"state":{"R3":false,"R3":true}}')
    def test_nonfinite_literals_and_numeric_overflow_are_rejected(self):
        for value in ['NaN','Infinity','-Infinity','1e999']:
            with self.subTest(value=value),self.assertRaises(v.VerificationError):
                v.decode_json('{"power":'+value+'}')
    def test_boolean_is_not_an_integer_counter(self):
        with self.assertRaises(v.VerificationError):v.zero_fields({'fits':False},['fits'],'fixture')
    def test_absent_or_true_scientific_gate_is_rejected(self):
        for x in [{},{'R3_accredited':True}]:
            with self.subTest(value=x),self.assertRaises(v.VerificationError):
                v.unchanged_false(x,['R3_accredited'],'fixture')


class DocumentaryClaimGuards(unittest.TestCase):
    def setUp(self):
        here=Path(__file__).resolve()
        source=next(p for p in [here.parent.parent,here.parent.parent.parent]
                    if (p/'guide_candidate/CANDIDATE_VIABILITY_AND_NEXT_DECISION.json').is_file())
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for name in ['methods_admissibility/METHOD_ADMISSIBILITY.json',
                     'methods_admissibility/THIRD_PROCEDURE_PLAN.json',
                     'guide_candidate/CANDIDATE_VIABILITY_AND_NEXT_DECISION.json',
                     'guide_candidate/REQUIREMENT_PRIMARY_EVIDENCE_GAP_GATE_MATRIX.json',
                     'guide_candidate/REQUIREMENT_PRIMARY_EVIDENCE_GAP_GATE_MATRIX.csv',
                     'guide_candidate/INPUT_BYTES_MANIFEST.json']:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/name,target)
    def tearDown(self):self.tmp.cleanup()
    def modify(self,name,change):
        path=self.root/name;value=json.loads(path.read_text());change(value)
        path.write_text(json.dumps(value))
    def test_actual_documentary_records_pass_without_execution(self):
        v.check_methods(self.root);self.assertGreaterEqual(v.check_guide_candidate(self.root),30)
    def test_unaccredited_three_methods_cannot_be_promoted(self):
        self.modify('methods_admissibility/METHOD_ADMISSIBILITY.json',
                    lambda d:d.update(confirmed_admissible_and_calibrated_primary_procedures=3))
        with self.assertRaises(v.VerificationError):v.check_methods(self.root)
    def test_MCS_cannot_supply_pairwise_p_values(self):
        self.modify('methods_admissibility/METHOD_ADMISSIBILITY.json',
                    lambda d:d['procedures'][2].update(pairwise_p_values_provided=True))
        with self.assertRaises(v.VerificationError):v.check_methods(self.root)
    def test_unselected_candidate_cannot_become_confirmatory(self):
        self.modify('guide_candidate/CANDIDATE_VIABILITY_AND_NEXT_DECISION.json',
                    lambda d:d.update(current_candidate_for_confirmation='MSE_only_controls'))
        with self.assertRaises(v.VerificationError):v.check_guide_candidate(self.root)
    def test_pending_defense_cannot_be_recoded_as_copying(self):
        self.modify('guide_candidate/CANDIDATE_VIABILITY_AND_NEXT_DECISION.json',
                    lambda d:d['R1_R2_R3_relation'].update(unobserved_student_defense_implies_copying_or_R2_no=True))
        with self.assertRaises(v.VerificationError):v.check_guide_candidate(self.root)
    def test_failed_V3_continuation_cannot_be_erased(self):
        self.modify('guide_candidate/CANDIDATE_VIABILITY_AND_NEXT_DECISION.json',
                    lambda d:d['candidate_rows'][1].update(development_decision='GO'))
        with self.assertRaises(v.VerificationError):v.check_guide_candidate(self.root)
    def test_corrupted_primary_source_binding_is_rejected(self):
        self.modify('guide_candidate/REQUIREMENT_PRIMARY_EVIDENCE_GAP_GATE_MATRIX.json',
                    lambda d:d['rows'][0].update(primary_sha256='0'*64))
        with self.assertRaises(v.VerificationError):v.check_guide_candidate(self.root)
    def test_missing_CSV_requirement_is_rejected(self):
        path=self.root/'guide_candidate/REQUIREMENT_PRIMARY_EVIDENCE_GAP_GATE_MATRIX.csv'
        lines=path.read_text().splitlines();path.write_text('\n'.join(lines[:-1])+'\n')
        with self.assertRaises(v.VerificationError):v.check_guide_candidate(self.root)


if __name__=='__main__':unittest.main()
