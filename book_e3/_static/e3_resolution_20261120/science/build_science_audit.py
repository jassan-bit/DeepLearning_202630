"""Audit saved planning artifacts and emit explicit prospective input gaps.

Only saved metadata, hashes and existing result rows are read. No scientific
resampling, p-value calculation, model fits, predictions or data downloads.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE='8d03a73d529eb3f9b2e89992b4488cfeb88a99e6'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_bytes())
def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def write_csv(path,data):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def build(root,out):
    out.mkdir(parents=True,exist_ok=True);power=root/'experiments/e3_completion_20261120/protocol_power'
    protocol=load(power/'protocol.DRAFT.json');registry=load(power/'planning_registry.json')
    science=load(root/'experiments/e3_science_followup_20261120/decision.json')
    source_receipt=load(root/'experiments/e3_science_followup_20261120/execution_receipt.json')
    old_inputs=load(root/'experiments/e3_science_followup_20261120/input_manifest.json')
    for item in old_inputs['files']:
        p=root/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    for relative,expected in registry['inputs'].items():assert sha(power/relative)==expected
    scenarios=registry['fixed_scenarios'];seed=rows(power/'results/seed_sensitivity.csv');calendar=rows(power/'results/calendar_sensitivity.csv');variance=rows(power/'results/variance_inputs.csv')
    assert len(seed)==6*3*3*3==162 and len(calendar)==6*3*2*3*3==324
    assert len({tuple(r[k] for k in ['rival','variance_reference','target_name','planned_gain']) for r in seed})==162
    assert len({tuple(r[k] for k in ['rival','HAC_Bartlett_lags','temporal_SD_multiplier','target_name','planned_gain']) for r in calendar})==324
    assert all(r['final_R_selected']=='False' and r['actual_three_test_power_established']=='False' for r in seed)
    assert all(r['reference_is_recommended_calendar_length']=='False' and r['guide_temporal_three_test_power_established']=='False' for r in calendar)
    family=protocol['family']['rivals'];assert len(family)==len(set(family))==19
    alias={'CNN-LSTM-atencion-aditiva E3':'additive_positive_head'};by_var={r['rival']:r for r in variance}
    family_rows=[]
    for index,rival in enumerate(family,1):
        known=by_var.get(alias.get(rival,rival));kind='neural' if index<=13 or index==17 else ('classic' if index<=16 else 'analytical')
        family_rows.append({'rival_id':f'M{index:02d}','rival':rival,'kind':kind,
            'saved_V3_pilot_variance_available':known is not None,
            'saved_pilot_seed_pairs':3 if known else None,
            'saved_rival_trajectories':int(known['rival_saved_trajectories']) if known else None,
            'saved_development_seed_SD':float(known['seed_SD']) if known else None,
            'scope':'exposed development55; not future confirmation variance' if known else 'not measured against V3 on its pilot calendar',
            'prospective_variance_sufficient':False,'new_evidence_needed':'paired planning trajectories, seed/time dependence and validated three-procedure power',
            'final_R':None})
    assert sum(r['saved_V3_pilot_variance_available'] for r in family_rows)==3
    write_csv(out/'main_family_variance_inputs.csv',family_rows)
    template=[]
    for panel in ['F00','F01','F02','F03']:
        for row in family_rows:
            template.append({'dataset_id':panel,'rival_id':row['rival_id'],'rival':row['rival'],
                'planning_seed_pairs':None,'paired_seed_SD':None,'temporal_long_run_variance':None,
                'seed_calendar_interaction_variance':None,'cross_rival_dependence_file_sha256':None,
                'planning_dates':None,'planning_predictions_sha256':None,'excluded_from_confirmation':None,
                'admission_and_access_log_sha256':None,'status':'NOT_MEASURED_NOT_ZERO'})
    write_csv(out/'future_variance_template.csv',template)
    input_paths=[root/'experiments/e3_science_followup_20261120'/n for n in ['decision.json','execution_receipt.json','input_manifest.json']]
    input_paths += [power/n for n in ['protocol.DRAFT.json','planning_registry.json','results/variance_inputs.csv','results/seed_sensitivity.csv','results/calendar_sensitivity.csv','results/joint_sensitivity.json','results/planning_receipt.json']]
    input_paths += [root/'experiments/e3_completion_20261120/reproduction/reference/additional_domain_admission_metadata.json',root/'experiments/e3_completion_20261120/reproduction/reference/admission.json',root/'data/e3_dashboard_v1/manifest.json',root/'data/e3_dashboard_v1/observations.csv']
    manifest={'kind':'post_result_read_only_input_provenance_not_preregistration','base_commit':BASE,
              'source_files':[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in input_paths],
              'older_saved_input_files_hash_verified':len(old_inputs['files']),'AI_assistance_declared':True}
    write_json(out/'input_provenance.json',manifest)
    audit={'kind':'executed_saved_scenario_audit','base_commit':BASE,'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'seed_scenarios':162,'calendar_scenarios':324,'total_scenarios':486,
        'seed_grid_formula':'6 controls × 3 variance references × 3 power targets × 3 alternative effects',
        'calendar_grid_formula':'6 controls × 3 HAC lags × 2 SD multipliers × 3 targets × 3 alternative effects',
        'main_rivals':19,'procedures_per_rival':3,'main_components':57,'control_variances':6,
        'main_rival_variances_available':3,'missing_main_rival_variances':16,'other_variances_are_separate_ablations':3,
        'pilot_seed_pairs':3,'guide_minimum_planning_seed_pairs':5,'minimum_confirmation_seed_pairs':10,
        'known_calendar_dates':55,'saved_conditional_R_range':[min(int(r['conditional_required_pairs_minimum10']) for r in seed),max(int(r['conditional_required_pairs_minimum10']) for r in seed)],
        'saved_conditional_date_reference_range':[min(float(r['asymptotic_reference_dates']) for r in calendar),max(float(r['asymptotic_reference_dates']) for r in calendar)],
        'ranges_are_not_recommendations':True,'inherited_delta':scenarios['BTC_inherited_RMSE_delta'],
        'delta_is_operationally_justified_for_V3':False,'alternative_minus_null_gap_definition':'Delta_alt - delta > 0; not the observed gain, not the failed2% continuation threshold',
        'joint_power_uses_independence_product_as_actual_estimate':False,'joint90_is_guide_mandatory':False,
        'final_R':None,'final_T':None,'actual_battery_power_established':False,'new_statistical_resamples':0,
        'new_p_values':0,'new_model_fits':0,'new_model_predictions':0,'GPU_seconds':0}
    write_json(out/'audit_486.json',audit)
    admitted=load(root/'experiments/e3_completion_20261120/reproduction/reference/additional_domain_admission_metadata.json')
    dashboard=load(root/'data/e3_dashboard_v1/manifest.json');observations=rows(root/'data/e3_dashboard_v1/observations.csv')
    ds=[{'id':'D00','instrument':'BTCUSDT Binance Spot','hash_scope':'cohort NPZ from accepted data manifest; not newly downloaded',
         'source':dashboard['sources']['cohort_npz'],'exposure':'training/validation/calibration/evaluation already exposed; V3 development uses2023–2024',
         'evaluation_dates':[min(r['target_date'] for r in observations if r['partition']=='evaluation'),max(r['target_date'] for r in observations if r['partition']=='evaluation')],
         'admitted_for_exploratory_task':True,'eligible_unexposed_confirmation':False}]
    reasons={'D01':'WIKI/AAPL archive ends2013; old test precedes V3 training; different source and trading clock do not restore blindness',
             'D02':'ETH/BTC crypto dependence;2021 history predates V3 BTC fit; old test exposed by additive audit and transfer',
             'D03':'same BTC underlying asD00, different contract/source but dependent;2024 test overlaps V3 fit/selection era and is exposed'}
    for key,value in admitted['datasets'].items():
        ds.append({'id':key,'source_metadata':value,'source_commit':admitted['source_commit'],
                   'raw_bytes_re_downloaded_in_this_audit':False,'hash_scope':'declared in accepted admission metadata; binding checked',
                   'admitted_for_exploratory_task':True,'eligible_unexposed_confirmation':False,'reason':reasons[key]})
    future=[]
    for key,domain,reason in [('F00','BTC spot future endpoints','main task; temporal shift only, not an additional independent market'),
                              ('F01','equity with a current OHLCV source','instrument class, session clock, corporate-action policy and volume units differ'),
                              ('F02','commodity futures with a frozen contract/roll rule','commodity exposure, contract volume, term structure and session definition differ'),
                              ('F03','rates instrument or rates ETF with traded volume','economic exposure, volatility/noise scale and instrument construction differ')]:
        future.append({'id':key,'purpose':'possible future panel metadata only; not a selected or admitted dataset',
                       'domain_to_assess':domain,'difference_axes_to_document':reason,'instrument':None,'source_URL':None,
                       'license':None,'version':None,'bytes_sha256':None,'first_endpoint':None,'last_endpoint':None,
                       'admitted_unexposed':False,'unopened_access_log_sha256':None,
                       'endpoint_rule':'future targets after a real protocol/data lock or a custodian-sealed never-opened panel; specify raw eight-close supports and seven-target boundary purge',
                       'exclusion_rule':'Do not relabel anyD00–D03 slice as unexposed or select domains by observed advantage',
                       'dependence_review':'Need economic/common-factor and calendar/underlying dependence assessment; different ticker/source is insufficient'})
    write_json(out/'data_eligibility.json',{'kind':'executed_exposure_audit_and_unadmitted_future_registry','known_panels':ds,'future_panels':future,
        'existing_unexposed_panels_certified':0,'future_raw_bytes_downloaded':0,
        'admission_required_fields':['source/license/version/hash','clock and volume units','OHLCV and target identity','two documented difference axes for each additional domain','corporate action/roll policy','split/purge absolute endpoints','author exposure and custodian access log'],
        'source_admission_metadata_sha256':sha(root/'experiments/e3_completion_20261120/reproduction/reference/additional_domain_admission_metadata.json')})
    methods=['P1_FUNCTIONAL_BLOCK_BOOTSTRAP','P2_RMSE_DELTA_HAC','P3_FUNCTIONAL_SUBSAMPLING']
    config={'schema_version':1,'kind':'FUTURE_PREPARATION_DRAFT_NOT_FROZEN_NOT_PREREGISTERED','base_commit':BASE,'candidate':'atom_crps',
        'science':{'decision':'NO_GO_CONFIRMATION_CURRENT_RECIPE','original_pilot_rule_preserved':True,'original_threshold':.02,
                   'atom_crps_error_increase_vs_atom_mse_only':science['adverse_pilot']['candidate_relative_error_increase_vs_best_control'],
                   'automatic_ablation_promotion':False,'renewed_rationale_sufficient':False,'new_mechanism_proposed':False},
        'estimand':{'estimator':'mean of individual seed RMSE differences on common target endpoints; not pooled or ensemble RMSE',
                    'future_population_target_proposal':'E_seed[sqrt(E_P error_rival^2)-sqrt(E_P error_V3^2)] for a predeclared temporal process P',
                    'null':'theta_j <= delta_j','alternative':'theta_j > delta_j','known55_calendar_reinterpretation_allowed':False,
                    'units':'raw unannualized population sigma7, ddof0','target_process_and_seed_sampling_reviewed':False},
        'family':{'rivals':family,'rival_count':19,'all_system_count':20,'procedures':methods,'components':57,
                  'Holm_family':'all57 primary p-values together at alpha0.05; require all3 adjustedp<0.05 perrival',
                  'separate_ablations':['atom_mse_only','fixed_probs','atom_plugin'],
                  'additional_domain_family_design':'predeclare main57 and additional171 adjusted families, or a stronger single228 family; evaluate error control for the final cross-domain decision before freeze',
                  'auxiliary_requirements':['MCS or critical-difference diagram','cross-dataset test with dependence justification','secondary metrics/subgroup/robustness families']},
        'margins':{'by_panel':{'F00':None,'F01':None,'F02':None,'F03':None},'old_BTC_reference':scenarios['BTC_inherited_RMSE_delta'],
                   'old_reference_approved_for_V3':False,'planning_alternative_gain_by_panel':None,
                   'rule':'Operationally justify delta before outcomes; specify Delta_alt > delta separately. Delta_alt=delta is a null boundary, not a90%-power alternative.'},
        'data':{'future_panels':future,'old_D00_D03_eligible':False,'minimum_additional_domains':3,'minimum_difference_axes_per_additional_domain':2},
        'treatment':{'window':7,'target_close_support':8,'forecast_horizon':1,'boundary_purge_targets_template':7,
                     'same_dates_preprocessing_fit_selection_calibration_all_models':True,'fit_only_scalers':True,
                     'HPO_template_configurations':3,'per_fit_seconds_template':120,'budget_templates_are_not_final':True,
                     'final_absolute_endpoints':None,'neural_systems_including_candidate':15,
                     'deterministic_controls':'one trajectory per recipe; no fake independent seeds or HPO runs',
                     'capacity':'report parameters and a defined FLOPs method; if needed matched-capacity benchmark fixed before outcomes',
                     'fairness_review_complete':False,'fixed_GPU_software_cohort_required':True},
        'sampling':{'planning_seed_pairs':3,'minimum_planning_pairs_per_stochastic_rival':5,
                    'complete_variances_for_four_panels_and19_rivals':False,'final_R':None,'minimum_final_R':10,'final_T':None,
                    'power_per_contrast_minimum':.9,'joint_report_required':True,'joint90_required':False,
                    'seed_x_date_rows_are_independent':False,'minimum_replicates_for_future_MC_precision_not_power_guarantee':10000},
        'battery':{'proposed_methods':methods,'reviewed_and_validated':False,
                   'old_GW_proposal_status':'not automatically a primary superiority test; omnibus conditional rejection does not establish the same one-sided RMSE margin',
                   'HLN_status':'optional finite-sample proposal requiring functional-specific validation; h1 differs from sigma7 overlap',
                   'implementation_and_method_receipts':None,'source_revision_is_future_draft_not_pilot_rule_change':True},
        'planning_simulation_contract':{'status':'PREPARED_NOT_EXECUTED','input_stage_required':'planning_simulation',
             'fit_DGP_only_on_admitted_planning_data_excluded_from_confirmation':True,
             'replicate_preserves':'same temporal path and crossed seed dependence for all rivals and all3 procedures',
             'null_validation':'boundary and below-margin nulls, partial-null configurations, shared shocks, heavy tails, weak/nonstationary alternatives; validate size/FWER before power',
             'one_dataset_per_summary_file':True,'pvalue_CSV_fields':['stage','dataset_id','replicate','R','T','rival','procedure','p_raw'],
             'marginal_decision':'intersection of all3 Holm-adjusted procedures perrival, not anysinglemethod',
             'joint_decision':'same-replicate intersection across19 rivals; do not multiply marginal powers',
             'Monte_Carlo_uncertainty':'one-sided simultaneous Clopper-Pearson exact-binomial bounds over19 rival intersections; joint bound separate; independent simulation replicates required; Monte Carlo precision is conditional on an unvalidated DGP'},
        'pilot':{'kind':'future separate planning-only pilot; no training launched here','data_partition_excluded_from_confirmation':True,
                 'seed_pairs_minimum':5,'fit_count_template':'4panels×15neurals×5seedfits=300 before any fresh HPO; count classic fits separately',
                 'approx_GPU_budget_reference_hours':10,'budget_reference_formula':'300×120s/3600, excludes setup, HPO, classics, failed attempts and analysis',
                 'available_days_and_GPU_reported_by_author':True,'availability_reconfirmation_required':False,
                 'why_not_run':'current scientific NO-GO, no admitted prospective planning/confirmation panels or validated battery; not lack of user willingness',
                 'incremental_old_pilot_extension':'adding2seeds to the old exposed pilot could improve conditional initialization SD but cannot fill16rivals or make55dates independent; not scheduled'},
        'freeze':{'annotated_tag_verified':False,'tag':None,'commit':None,'date':None,'outcome_access_order_proof':None},
        'evidence_bindings':{},'confirmation_Colab_generated':False,'new_protocol_tag_created':False,
        'R3_accredited':False,'AI_assistance_declared':True}
    write_json(out/'prospective_design.DRAFT.json',config)
    write_json(out/'scientific_decision.json',{'kind':'new_executed_documentary_and_contract_resolution','base_commit':BASE,
       'scientific_decision':science['scientific_decision'],'R3_accredited':False,
       'preserved_additive_jobs':796,'preserved_V3_fits':30,'preserved_transfer_trajectories':51,'preserved_transfer_rows':20400,
       'candidate_beats_best_transfer_control_domains':0,'transfer_domains':3,
       'pilot_error_increase':science['adverse_pilot']['candidate_relative_error_increase_vs_best_control'],
       'rule_changed':False,'ablation_promoted':False,'new_model_fits':0,'new_predictions':0,
       'what_closed':['all486 saved scenario identities and flags audited','all19 main rivals and16missing variance inputs enumerated','D00–D03 exposure and accepted provenance documented','future power aggregation/preflight contract executable'],
       'what_remains':['renewed scientific rationale','admitted unexposed four-panel design','operational margins/alternative gain justification','complete seed/time covariance','three-method null-calibration and power validation','finalR/T and annotated prospective registration']})
    return {'status':'executed_saved_science_audit_and_future_contract_preparation','input_files_hash_verified':len(old_inputs['files'])+len(registry['inputs']),
            'scenario_rows':486,'main_family_rows':19,'future_variance_template_rows':76,
            'current_data_unexposed_eligible':0,'new_market_data_downloads':0,'new_model_fits':0,
            'new_model_predictions':0,'new_statistical_resamples':0,'confirmation_go':False,'R3_accredited':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=build(args.root.resolve(),args.output.resolve());write_json(args.output/'execution_receipt.json',report);print(json.dumps(report))
