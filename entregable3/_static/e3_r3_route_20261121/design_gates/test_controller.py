"""Synthetic metadata only, created in TemporaryDirectory; no scientific run."""
import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import controller as c


REPO = next((p for p in [Path(os.environ.get('R3_REPO_ROOT', '.')), c.HERE.parents[2],
                         Path('/workspace/e3_completion_worktree')]
             if (p/'experiments/e3_statistics_remediation_20261121/guide_power/power_inputs_audit.py').is_file()),
            Path(os.environ.get('R3_REPO_ROOT', '.')))


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.serial = 0
        module, base = c.inherited(REPO)
        self.target = {k: base[k] for k in ('estimand','units','seed_uncertainty','calendar_uncertainty')}
        artifact = self.put({'synthetic_candidate': 'test-only'})
        candidate = {'version':'synthetic-only', 'artifact':artifact}
        rivals = [{'id':f'r{i}', 'kind':'stochastic' if i < 10 else 'deterministic',
                   'recipe':self.put({'synthetic_recipe':i})} for i in range(11)]
        self.rival_ids = [r['id'] for r in rivals]
        methods = []
        for i in range(3):
            mid = f'p{i}'
            methods.append({'id':mid, 'global_or_multiple':i == 0,
                'implementation':self.put({'synthetic_method_implementation':i}),
                'definition':self.put({'method_id':mid, 'estimand':base['estimand'],
                    'sampling_population':[base['seed_uncertainty'],base['calendar_uncertainty']],
                    'null':'synthetic equality', 'direction':'candidate_better', 'assumptions':'synthetic reviewed assumptions'}),
                'contrast_mapping':self.put({'method_id':mid, 'rival_ids':self.rival_ids,
                    'translation':'direct_contrast','derivation':'synthetic-only test derivation',
                    'component_keys_by_panel_rival':{f'F{p:02d}|{r}':[f'F{p:02d}|{r}|{mid}']
                        for p in range(4) for r in self.rival_ids}})})
        panels = [{'id':f'F{i:02d}', 'task':'forecast_unannualized_sigma7', 'forecast_horizon':1,
                   'target_close_support':8, 'planning_dates':['2027-01-09','2027-01-10'],
                   'confirmation_dates':['2027-02-09','2027-02-10'], 'practical_margin':.1,
                   'planning_alternative':.2, 'margin_units':base['units'],
                   'margin_rationale':'synthetic input, not scientific justification'} for i in range(4)]
        panel_data={p['id']:{part:self.put({'synthetic':p['id']+part})
                            for part in ('planning','confirmation')} for p in panels}
        for p in panels:
            p['data_sha256']={part:binding['sha256'] for part,binding in panel_data[p['id']].items()}
        components = sorted(p['id']+'|'+r['id']+'|'+m['id'] for p in panels for r in rivals for m in methods)
        rule_code = self.put({'synthetic_full_rule_implementation':1})
        rule = self.put({'implementation':rule_code, 'method_ids':[m['id'] for m in methods], 'component_keys':components,'contrast_method_keys':components,
                        'requires_all_three':True,'includes_direction_margin_effect_CI_adequacy_robustness':True,
                        'complete_rule_error_control':{'scope':'complete_decision_rule','alpha':.05,
                            'event':'any_false_rejection_complete_declared_family','justification':'synthetic primary rule'}})
        family = {'rivals':rivals, 'top10':self.rival_ids[:10], 'strong_baselines':self.rival_ids[10:],
                  'other_declared_rivals':[], 'multiplicity_groups':[{'id':'primary','components':components,
                      'adjustment':'Holm','alpha':.05,'justification':'synthetic joint family'}],
                  'decision_rule':rule}
        self.design = {'candidate':candidate,'family':family,'methods':methods,'panels':panels,
                       'sampling':{'planning_seed_ids':list(range(5)), 'final_seed_ids':list(range(100,110)),
                                   'final_R':10,'final_T':{p['id']:2 for p in panels},'seed_x_date_IID':False},
                       'finite_calendar_bias_statement':'Synthetic scope declares finite-T bias; no extra-seed cure.',
                       'resources':{'hardware':'synthetic CPU','software_environment_sha256':'a'*64,
                                    'max_fit_attempts':500,'max_wall_seconds':10000,
                                    'per_fit_cap_seconds':{x:1 for x in ['synthetic-only']+self.rival_ids},
                                    'additional_HPO_fit_jobs':0},
                       'persistence':{'verified_checkpoint_required':True,'skip_verified_complete':True,
                                      'overwrite_existing':False,'storage_uri':'test://temporary', 'atomic_write':True}}
        self.config = {'kind':'SYNTHETIC_SOFTWARE_TEST_NOT_SCIENTIFIC_EVIDENCE', 'design':self.design,
                       'evidence':{},'campaign_authorization':None}
        self.evidence = {}
        for name in ('r1','r2'):
            self.evidence[name] = self.receipt(decision='accredited_for_this_candidate',reviewer='synthetic reviewer',
                external_evidence_uri='https://synthetic.invalid/review',reviewed_at='2026-12-01T00:00:00Z')
        self.evidence['candidate_decision'] = self.receipt(planning_allowed=True,recipe_abandoned=False)
        admissions = []
        for p in panels:
            admissions.append({'id':p['id'], 'license':'synthetic','source_uri':'https://synthetic.invalid/data',
                'exposure_declaration_uri':'https://synthetic.invalid/exposure','calendar':'synthetic calendar',
                'volume_units':'synthetic units','dependence_review':'synthetic','difference_axes':['economic','session'],
                'planning_support':['2027-01-01','2027-01-10'],'confirmation_support':['2027-02-01','2027-02-10'],
                'planning_data':panel_data[p['id']]['planning'],
                'confirmation_data':panel_data[p['id']]['confirmation'],
                'planning_partition_used_in_other_phase':False,'confirmation_partition_used_in_other_phase':False,
                'planning_first_outcome_access_at':'2027-01-11T00:00:00Z',
                'confirmation_first_outcome_access_at':'2027-02-11T00:00:00Z'})
        self.evidence['admission'] = self.receipt(admitted_panel_ids=[p['id'] for p in panels],panels=admissions)
        frozen = self.put({'registered_at':'2026-12-01T00:00:00Z',
            'external_registration_uri':'https://synthetic.invalid/calibration', 'MC_error_probability':.05,
            'decision_rule_sha256':rule['sha256'],'family_sha256':c.sha(family),
            'acceptance_criteria':{
                'FWER':{'scope':'complete_decision_rule','decision_rule_sha256':rule['sha256'],
                        'nominal_alpha':.05,'maximum_error_rate':.05},
                'primary':{'scope':'multiplicity_group','group_id':'primary',
                           'nominal_alpha':.05,'maximum_error_rate':.05}}})
        scenarios = ['complete_null','partial_null','seed_only','calendar_only','shared_shocks',
                     'heavy_tails','degeneracy','heterogeneous_dependence_population_equal']
        upper = .01 + math.sqrt(math.log(2*len(scenarios)/.05)/(2*10000))
        self.evidence['calibration'] = self.receipt(decision_rule_sha256=rule['sha256'],
            decision_rule_implementation_sha256=rule_code['sha256'],
            method_definition_sha256s=[m['definition']['sha256'] for m in methods],
            method_implementation_sha256s=[m['implementation']['sha256'] for m in methods],
            validated_R_T_scope={'R_range':[5,10],'T_range':[2,2],
                'population_null_and_finite_calendar_target':'synthetic scope',
                'restrictions_and_extrapolation_review':'synthetic-only'},
            frozen_validation_protocol=frozen, first_validation_result_at='2026-12-02T00:00:00Z',
            strong_control_rows=[{'scenario':s,'criterion_id':criterion,'scope':scope,'group_id':group,
                'independent_replicates':10000,'false_rejection_replicates':100,'event':event,
                'simultaneous_MC_upper':upper} for s in scenarios for criterion,scope,group,event in (
                    ('FWER','complete_decision_rule',None,'any_false_rejection_complete_declared_family'),
                    ('primary','multiplicity_group','primary','any_false_rejection_in_multiplicity_group'))],
            reviewer='synthetic',external_review_uri='https://synthetic.invalid/calibration-review',
            finite_calendar_bias_scope='synthetic documented population-equality scope')
        archived = self.put(self.design)
        for phase in ('planning','confirmation'):
            self.evidence[phase+'_registration'] = self.receipt(archived_design=archived,
                external_registration_uri='https://synthetic.invalid/registration',immutable_revision='synthetic0001',
                custodian='synthetic',registered_at='2026-12-03T00:00:00Z',reviewed_at='2026-12-04T00:00:00Z',
                external_review_uri='https://synthetic.invalid/order-review')
        self.evidence['runner_review'] = {'design_sha256':c.sha(self.design),'reviewer':'synthetic',
            'external_evidence_uri':'https://synthetic.invalid/runner','runner_code':self.put({'synthetic_runner_source':1}),
            'enforces_wall_and_attempt_caps':True,'checkpoint_integrity_and_resume_tested':True}
        observations = []
        order = [p['id']+'|'+r for p in panels for r in self.rival_ids]
        paired = {key:[] for key in order}
        for p in panels:
            for ri,r in enumerate(rivals):
                for seed in range(5):
                    rows = []
                    for di,day in enumerate(p['planning_dates']):
                        rival_error = (1+ri*.02+seed*.01+di*.01)**2 if r['kind']=='stochastic' else 4.
                        row={'panel':p['id'],'rival':r['id'],'seed':seed,'date':day,
                             'candidate_squared_error':.25,'rival_squared_error':rival_error}
                        observations.append(row)
                        rows.append(row)
                    paired[p['id']+'|'+r['id']].append(math.sqrt(sum(x['rival_squared_error'] for x in rows)/2)-.5)
        seed_cov=[]
        for a in order:
            row=[]
            for b in order:
                aa,bb=paired[a],paired[b]
                ma,mb=sum(aa)/5,sum(bb)/5
                row.append(sum((x-ma)*(y-mb) for x,y in zip(aa,bb))/4)
            seed_cov.append(row)
        zero=[[0. for _ in order] for _ in order]
        self.cube = {'matrix_order':order,'variance_units':base['units']+'^2',
            'planning_seed_ids':list(range(5)),'dates_by_panel':{p['id']:p['planning_dates'] for p in panels},
            'date_seed_cells_are_IID':False,'paired_observations':self.put({'rows':observations}),
            'seed_covariance':seed_cov, 'calendar_LRV':copy.deepcopy(zero),
            'seed_calendar_interaction_LRV':copy.deepcopy(zero),'numerical_relative_tolerance':1e-10,
            'temporal_LRV_derivation':'synthetic-only scope, not an estimator validation',
            'cross_rival_and_cross_panel_dependence':'synthetic full matrices',
            'finite_T_estimand_bias_assessment':'synthetic finite-T scope'}
        self.evidence['paired_covariance']=self.receipt(crossed_covariance_payload=self.put(self.cube))
        contrasts={key:{'procedure_rejects':{'p0':True,'p1':True,'p2':True},
                        'remaining_complete_rule_pass':True} for key in order}
        self.power_rows={'replicates':[{'replicate':i,'contrasts':copy.deepcopy(contrasts)} for i in range(400)]}
        self.evidence['power']=self.receipt(stage='prospective_planning_simulation',
            uses_observed_confirmation_results=False,decision_rule_sha256=rule['sha256'],selected_R=10,
            decision_rule_implementation_sha256=rule_code['sha256'],
            selected_T={p['id']:2 for p in panels},independent_replicates_review_uri='https://synthetic.invalid/independence',
            DGP_scope_and_finite_calendar_bias_review_uri='https://synthetic.invalid/DGP',
            replicate_decisions=self.put(self.power_rows),MC_error_probability=.05)
        self.auth={'design_sha256':c.sha(self.design),'phase':'confirmation','authorizer':'synthetic',
                   'external_authorization_uri':'https://synthetic.invalid/budget','max_fit_attempts':500,
                   'max_wall_seconds':10000}
        self.flush()

    def put(self, value):
        self.serial+=1
        path=self.root/f'{self.serial}.json'
        path.write_bytes(c.canonical(value))
        return {'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def receipt(self, **extra):
        return dict(self.target, candidate_version=self.design['candidate']['version'],
                    candidate_sha256=self.design['candidate']['artifact']['sha256'],
                    design_sha256=c.sha(self.design),family_sha256=c.sha(self.design['family']),**extra)

    def flush(self):
        self.config['evidence']={k:self.put(v) for k,v in self.evidence.items()}
        self.config['campaign_authorization']=self.put(self.auth)

    def assess(self, phase='confirmation'):
        self.flush()
        return c.assess(self.config,REPO,self.root,phase)

    def mixed_level_family(self):
        """One mapped global level.10 plus primary Holm.05; complete rule stays.05."""
        method=self.design['methods'][0]
        mapping=json.loads((self.root/method['contrast_mapping']['path']).read_text())
        mapping.update(translation='justified_global_to_contrast',global_p_repeated_as_contrast=False,
                       directional_exclusion_construction='synthetic validated exclusion construction')
        mapping['component_keys_by_panel_rival']={key:[key.split('|')[0]+'|global_p0']
                                                 for key in mapping['component_keys_by_panel_rival']}
        method['contrast_mapping']=self.put(mapping)
        family=self.design['family']
        primary=[key for key in family['multiplicity_groups'][0]['components'] if not key.endswith('|p0')]
        global_keys=sorted({keys[0] for keys in mapping['component_keys_by_panel_rival'].values()})
        family['multiplicity_groups']=[{'id':'primary','components':primary,'adjustment':'Holm',
            'alpha':.05,'justification':'synthetic primary'}, {'id':'global','components':global_keys,
            'adjustment':'intrinsic_global','alpha':.10,'justification':'synthetic explicitly declared global role'}]
        rule=json.loads((self.root/family['decision_rule']['path']).read_text())
        rule['component_keys']=sorted(primary+global_keys)
        family['decision_rule']=self.put(rule)
        calibration=self.evidence['calibration']
        calibration['decision_rule_sha256']=family['decision_rule']['sha256']
        self.evidence['power']['decision_rule_sha256']=family['decision_rule']['sha256']
        protocol=json.loads((self.root/calibration['frozen_validation_protocol']['path']).read_text())
        protocol.update(decision_rule_sha256=family['decision_rule']['sha256'],family_sha256=c.sha(family))
        protocol['acceptance_criteria']['FWER']['decision_rule_sha256']=family['decision_rule']['sha256']
        protocol['acceptance_criteria']['global']={'scope':'multiplicity_group','group_id':'global',
                                                  'nominal_alpha':.10,'maximum_error_rate':.10}
        for row in list(calibration['strong_control_rows']):
            if row['criterion_id']=='primary':
                calibration['strong_control_rows'].append(dict(row,criterion_id='global',group_id='global'))
        nrows=len(calibration['strong_control_rows'])
        for row in calibration['strong_control_rows']:
            row['simultaneous_MC_upper']=.01+math.sqrt(math.log(nrows/.05)/(2*10000))
        calibration['frozen_validation_protocol']=self.put(protocol)
        design_hash,family_hash=c.sha(self.design),c.sha(family)
        for value in self.evidence.values():
            if 'design_sha256' in value:value['design_sha256']=design_hash
            if 'family_sha256' in value:value['family_sha256']=family_hash
        for phase in ('planning','confirmation'):
            self.evidence[phase+'_registration']['archived_design']=self.put(self.design)
        self.auth['design_sha256']=design_hash


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)

    def blocked(self, result, gate):
        self.assertFalse(result['technical_preflight_ready'])
        self.assertFalse(result['execution_permit_conditionally_ready'])
        self.assertFalse(result['R3_accredited'])
        self.assertIn(gate,result['blockers'])
        self.assertEqual([],result['execution_manifest'])

    def test_fully_consistent_synthetic_fixture_has_conditional_permit_never_accreditation(self):
        x=self.f.assess()
        self.assertTrue(x['technical_preflight_ready'],x['errors'])
        self.assertFalse(x['R3_accredited'])
        self.assertFalse(x['power_certified_by_this_program'])
        self.assertFalse(x['receipt_authenticity_verified_by_program'])
        self.assertEqual(444,len(x['execution_manifest']))
        self.assertEqual('NOT_ASSESSED_BY_SOFTWARE',x['scientific_accreditation'])

    def test_current_draft_blocks_and_keeps_R_T_and_family_unset(self):
        draft=json.loads((c.HERE/'protocol.DRAFT.json').read_text())
        x=c.assess(draft,REPO,self.tmp.name)
        self.assertFalse(x['technical_preflight_ready'])
        self.assertEqual(9,len(x['blockers']))
        self.assertIsNone(draft['design']['family'])
        self.assertIsNone(draft['design']['sampling']['final_R'])

    def test_planning_permit_does_not_require_final_power_or_covariance(self):
        self.f.auth['phase']='planning'
        del self.f.evidence['power']
        del self.f.evidence['paired_covariance']
        self.f.design['sampling']['final_R']=None
        self.f.design['sampling']['final_T']=None
        self.f.design['sampling']['final_seed_ids']=None
        # Rebuild external scope records to the deliberately changed design.
        for key,value in self.f.evidence.items():
            if 'design_sha256' in value:value['design_sha256']=c.sha(self.f.design)
        self.f.auth['design_sha256']=c.sha(self.f.design)
        self.f.evidence['planning_registration']['archived_design']=self.f.put(self.f.design)
        x=self.f.assess('planning')
        self.assertTrue(x['technical_preflight_ready'],x['errors'])
        self.assertEqual(224,len(x['execution_manifest']))

    def test_own_R1_cannot_be_inherited(self):
        self.f.evidence['r1']['candidate_sha256']='0'*64
        self.blocked(self.f.assess(),'candidate_own_R1_R2_and_continuation')

    def test_abandoned_candidate_blocks(self):
        self.f.evidence['candidate_decision']['recipe_abandoned']=True
        self.blocked(self.f.assess(),'candidate_own_R1_R2_and_continuation')

    def test_bool_float_or_duplicate_seeds_fail(self):
        for value in ([False,1,2,3,4],[0.,1,2,3,4],[0,1,2,3,3]):
            with self.subTest(value=value):
                saved=self.f.design['sampling']['planning_seed_ids']
                self.f.design['sampling']['planning_seed_ids']=value
                self.blocked(self.f.assess(),'estimand_margin_and_seed_sampling')
                self.f.design['sampling']['planning_seed_ids']=saved

    def test_family_omission_and_repeated_global_mapping_fail(self):
        self.f.design['family']['multiplicity_groups'][0]['components'].pop()
        self.blocked(self.f.assess(),'exact_family_three_methods_and_mapping')

    def test_repeating_one_global_raw_p_as_direct_contrasts_fails(self):
        method=self.f.design['methods'][0]
        mapping=json.loads((self.f.root/method['contrast_mapping']['path']).read_text())
        mapping['component_keys_by_panel_rival']={key:['one_global_p']
             for key in mapping['component_keys_by_panel_rival']}
        method['contrast_mapping']=self.f.put(mapping)
        self.blocked(self.f.assess(),'exact_family_three_methods_and_mapping')

    def test_overlap_in_raw_target_supports_fails(self):
        self.f.evidence['admission']['panels'][0]['confirmation_support'][0]='2027-01-10'
        self.blocked(self.f.assess(),'four_admitted_disjoint_panels')

    def test_delta_alternative_units_and_MSE_drift_fail(self):
        self.f.design['panels'][0]['planning_alternative']=.1
        self.blocked(self.f.assess(),'estimand_margin_and_seed_sampling')
        self.f.design['panels'][0]['planning_alternative']=.2
        self.f.design['target']={'estimand':'mean_MSE'}
        self.blocked(self.f.assess(),'estimand_margin_and_seed_sampling')

    def test_three_exploratory_screens_are_not_strong_control(self):
        self.f.evidence['calibration']['strong_control_rows']=self.f.evidence['calibration']['strong_control_rows'][:3]
        self.blocked(self.f.assess(),'strong_calibration_for_exact_rule')

    def test_calibration_upper_bound_must_match_counts(self):
        self.f.evidence['calibration']['strong_control_rows'][0]['false_rejection_replicates']=900
        self.blocked(self.f.assess(),'strong_calibration_for_exact_rule')

    def test_global_10_level_cannot_relax_complete_primary_05_level(self):
        self.f.mixed_level_family()
        accepted=self.f.assess()
        self.assertTrue(accepted['technical_preflight_ready'],accepted['errors'])
        calibration=self.f.evidence['calibration']
        protocol=json.loads((self.f.root/calibration['frozen_validation_protocol']['path']).read_text())
        protocol['acceptance_criteria']['FWER'].update(nominal_alpha=.10,maximum_error_rate=.10)
        calibration['frozen_validation_protocol']=self.f.put(protocol)
        self.blocked(self.f.assess(),'strong_calibration_for_exact_rule')

    def test_each_group_requires_its_own_calibration_scope_and_level(self):
        self.f.mixed_level_family()
        calibration=self.f.evidence['calibration']
        calibration['strong_control_rows']=[r for r in calibration['strong_control_rows']
                                            if r['criterion_id']!='global']
        for row in calibration['strong_control_rows']:
            row['simultaneous_MC_upper']=.01+math.sqrt(math.log(len(calibration['strong_control_rows'])/.05)/(2*10000))
        self.blocked(self.f.assess(),'strong_calibration_for_exact_rule')

    def test_local_registration_after_access_fails(self):
        self.f.evidence['confirmation_registration']['registered_at']='2027-03-01T00:00:00Z'
        self.blocked(self.f.assess(),'real_registration_before_outcome_access')

    def test_covariance_hash_cannot_be_merely_64_characters(self):
        self.f.evidence['paired_covariance']['crossed_covariance_payload']['sha256']='f'*64
        self.blocked(self.f.assess(),'complete_covariance_and_joint_R_T_design')

    def test_replacing_admitted_data_after_design_freeze_fails(self):
        self.f.evidence['admission']['panels'][0]['confirmation_data']=self.f.put({'replacement':'synthetic'})
        self.blocked(self.f.assess(),'four_admitted_disjoint_panels')

    def test_method_implementation_bytes_cannot_change(self):
        binding=self.f.design['methods'][0]['implementation']
        (self.f.root/binding['path']).write_text('changed synthetic implementation')
        self.blocked(self.f.assess(),'exact_family_three_methods_and_mapping')

    def test_asymmetric_or_indefinite_covariance_fails(self):
        self.f.cube['calendar_LRV'][0][1]=1
        self.f.evidence['paired_covariance']['crossed_covariance_payload']=self.f.put(self.f.cube)
        self.blocked(self.f.assess(),'complete_covariance_and_joint_R_T_design')
        with self.assertRaises(ValueError):c.positive_semidefinite([[1,2],[2,1]],2,1e-10)

    def test_covariance_must_match_saved_seed_scores(self):
        self.f.cube['seed_covariance']=[[0. for _ in range(44)] for _ in range(44)]
        self.f.evidence['paired_covariance']['crossed_covariance_payload']=self.f.put(self.f.cube)
        self.blocked(self.f.assess(),'complete_covariance_and_joint_R_T_design')

    def test_missing_source_cube_cell_is_not_imputed(self):
        source=json.loads((self.f.root/self.f.cube['paired_observations']['path']).read_text())
        source['rows'].pop()
        self.f.cube['paired_observations']=self.f.put(source)
        self.f.evidence['paired_covariance']['crossed_covariance_payload']=self.f.put(self.f.cube)
        self.blocked(self.f.assess(),'complete_covariance_and_joint_R_T_design')

    def test_T_must_match_actual_calendar_without_arbitrary_32_floor(self):
        self.assertTrue(self.f.assess()['technical_preflight_ready'])
        self.f.design['sampling']['final_T']['F00']=3
        self.blocked(self.f.assess(),'complete_covariance_and_joint_R_T_design')

    def test_power_requires_intersection_of_all_three_and_remaining_rule(self):
        for row in self.f.power_rows['replicates']:
            row['contrasts']['F00|r0']['procedure_rejects']['p2']=False
        self.f.evidence['power']['replicate_decisions']=self.f.put(self.f.power_rows)
        self.blocked(self.f.assess(),'complete_rule_power_and_joint_report')

    def test_observed_power_is_forbidden(self):
        self.f.evidence['power']['uses_observed_confirmation_results']=True
        self.blocked(self.f.assess(),'complete_rule_power_and_joint_report')

    def test_joint_power_below_90_is_reported_without_universal_90_gate(self):
        keys=list(self.f.power_rows['replicates'][0]['contrasts'])
        for i,key in enumerate(keys):
            for row_index in range(3*i,3*i+3):
                self.f.power_rows['replicates'][row_index]['contrasts'][key]['remaining_complete_rule_pass']=False
        self.f.evidence['power']['replicate_decisions']=self.f.put(self.f.power_rows)
        x=self.f.assess()
        self.assertTrue(x['technical_preflight_ready'],x['errors'])
        self.assertLess(x['saved_power_summary']['joint_power'],.9)
        self.assertFalse(x['saved_power_summary']['joint90_required'])

    def test_method_implementation_and_joint_R_T_calibration_scope_must_match(self):
        self.f.evidence['calibration']['validated_R_T_scope']['T_range']=[3,20]
        self.blocked(self.f.assess(),'strong_calibration_for_exact_rule')

    def test_budget_authorization_and_integrity_required(self):
        self.f.auth['max_fit_attempts']=10
        self.blocked(self.f.assess(),'authorized_resources_and_persistent_runner')

    def test_symlinks_and_path_escape_fail(self):
        old,_=c.inherited(REPO)
        (self.f.root/'link').symlink_to('/etc/hosts')
        for path in ('link','../outside.json','/etc/hosts'):
            with self.assertRaises(ValueError):old.safe_file(self.f.root,path)

    def test_checkpoint_skip_only_verified_exact_job_and_artifacts(self):
        x=self.f.assess()
        job=x['execution_manifest'][0]
        artifact=self.f.put({'synthetic_completed_job':1})
        receipt={'job_id':job['job_id'],'design_sha256':job['design_sha256'],
                 'status':'complete','artifacts':[artifact]}
        roots={'repo_root':REPO,'artifact_root':self.f.root}
        self.assertEqual('SKIP_VERIFIED_COMPLETE',c.checkpoint_state(job,receipt,roots))
        wrong=dict(receipt,job_id='wrong')
        with self.assertRaises(ValueError):c.checkpoint_state(job,wrong,roots)
        (self.f.root/artifact['path']).write_text('corrupted')
        with self.assertRaises(ValueError):c.checkpoint_state(job,receipt,roots)


if __name__=='__main__':unittest.main()
