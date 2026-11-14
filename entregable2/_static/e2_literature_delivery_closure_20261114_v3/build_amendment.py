"""Rebuild the dated literature decisions and arithmetic scenarios from receipts."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
OLD = ROOT/'references/literature_e2/delivery_closure_20261114_v2'


def load(path):
    return json.loads(path.read_text())


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def table(path, values):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n')
        writer.writeheader()
        for value in values:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in value.items()})


quality=read(B/'quality/current_nine_quality_before_data_access.csv')
data_access={r['study_id']:r for r in read(B/'quality/current_nine_data_access.csv')}
for row in quality:
    access=data_access[row['study_id']]
    row['public_data_availability_status']=access['public_data_availability_status']
    row['public_data_availability_evidence']=access['official_url']+'; '+access['dataset_release_mapping']+'; exact experimental snapshot/preprocessing not verified'
table(B/'quality/current_nine_quality.csv',quality)


decisions = load(OLD/'eligibility/applied_target_decisions.json')
updates = load(B/'primary_reviews/pair_c_primary_assessments.json')['studies']
updates += load(B/'primary_reviews/pair_d_adjudications.json')['studies']
updates = {v['doi'].casefold():v for v in updates}
for row in decisions:
    new=updates.get(row['doi'].casefold())
    row['prior_decision']=row['decision']
    row['new_manuscript_reading_claimed']=False
    if new is None:
        if row['decision']=='contexto_elegibilidad_pendiente':
            row['new_manuscript_reading_claimed']=True
            row['decision_stage']='relectura_primaria_focal_constructo_no_resuelto'
            row['evidence_pointers'].append('primary_reviews/two_constructs.md#doi='+row['doi'])
        continue
    included=new.get('decision')=='incluido'
    row['decision']=row['applied_eligibility']='incluido' if included else 'excluido'
    row['decision_reason']=new.get('decision_reason',new.get('reason'))
    row['adjudicated_source_role']=new.get('adjudicated_source_role','contexto_aplicado_otra_poblacion')
    row['narrative_stratum']=new.get('narrative_stratum','volatilidad_realizada_acciones')
    row['decision_stage']='lectura_cuerpo_editorial_focal' if included else 'cribado_poblacion_resumen_fragmento_editorial'
    access=new.get('access',{})
    row['assessed_primary_sections_available']=bool(included)
    row['new_manuscript_reading_claimed']=bool(included)
    row['source_url']=access.get('source_url',new.get('publisher_url',new.get('source_url')))
    row['source_version']=access.get('source_version',new.get('source_version'))
    row['source_locators']=access.get('locators',new.get('indexed_partial_extraction',{}).get('locators',[new.get('abstract_locator')]))
    row['evidence_pointers'].append('primary_reviews/'+('pair_c_primary_assessments.json' if 'access' in new else 'pair_d_adjudications.json')+'#doi='+row['doi'])
    if included:
        row['target_extracted']=new['target']['formula']
        row['neural_architecture_extracted']=new['architecture']['proposed']
        row['reported_result']=new['reported_results']
        row['reporting_and_bias_limits']=new['leakage_and_fairness']['limits']
    else:
        row['target_extracted']=new.get('target',{}).get('construct',new.get('abstract_partial_extraction',{}).get('objective','Volatilidad realizada bursátil; receta completa no verificada'))
        row['reported_result']=None
        row['reporting_and_bias_limits']=['Exclusión del núcleo aplicado cripto por población; contexto de mecanismo conservado. No equivale a exclusión de toda la revisión de mecanismos.','Cuerpo completo no recuperado; no se atribuyen resultados numéricos a una nueva lectura primaria.']
save(B/'eligibility.json',decisions)
table(B/'eligibility.csv',decisions)
counts=dict(Counter(r['decision'] for r in decisions))
save(B/'focal_flow.json',{'focal_targets':30,'prior_dispositions':{'incluido':13,'excluido':7,'no_recuperado':8,'contexto_elegibilidad_pendiente':2},'current_dispositions':counts,'new_primary_body_readings_for_previous_unrecovered_targets':1,'new_population_exclusions_at_screening':3,'context_constructs_rechecked_but_not_resolved':2,'global_received_groups_historical':9559,'global_PRISMA_closed':False,'new_1104_search_is_separate_identification_round':True,'new_search_records_screened_individually':None,'new_search_fulltext_included':None,'global_eligible':None,'global_included':None,'human_independent_double_review':False})

# Same calendar-age policy and fixed denominator as the preserved C; all counts
# below come from one newly supplied Scopus export, with no version summation.
source_map = {
 'FCN-1D':('2-s2.0-85030974623',2016),
 'ResNet-1D':('2-s2.0-85030974623',2016),
 'Inception-1D':('2-s2.0-85090433635',2019),
 'LSTM-autoencoder con predictor':('2-s2.0-85076460928',2019),
 'TimesNet':('2-s2.0-85196367497',2022),
 'TSMixer':('2-s2.0-86000299816',2023),
 'TiDE':('2-s2.0-86000040350',2023),
 'TimeMixer':('2-s2.0-85186508140',2024),
 'PatchTST':('2-s2.0-85192521637',2022),
 'iTransformer':('2-s2.0-85200562438',2023),
}
snapshot={r['EID']:r for r in read(B/'citation_snapshot/records.csv')}
previous=read(OLD/'citation_score/current10_C_status_v2.csv')
components=[]
for old in previous:
    eid,year=source_map[old['model']]
    source=snapshot[eid]
    citations=int(source['Cited by']);age=2026-year+1
    components.append({'model':old['model'],'study_id':old['study_id'],'EID':eid,'DOI_in_new_export':source['DOI'],'title_in_new_export':source['Title'],'authors_in_new_export':source['Authors'],'publication_year_in_export':source['Year'],'earliest_study_year':year,'citation_age_years':age,'citations':citations,'C_new':min(1.,max(0.,citations/age/487.)),'C_preserved':old['C_preserved'],'citation_count_preserved':old['citation_count_preserved'],'new_snapshot_date':'2026-11-14','provider':'Scopus','old_C_replaced':False,'catalogue_changed':False})
table(B/'citation_snapshot/model_components.csv',components)

# P9 is unchanged. The author-artifact extension is a separate, explicitly
# retrospective scenario; it never attributes ensemble predictions to one net.
p9=[r for r in read(OLD/'citation_score/conditional_score_by_scenario.csv') if r['scenario']=='posthoc_intrapaper_9_primary']
candidate=load(B/'inception_single/candidate.json')
assert candidate['candidate']['P']==0 and candidate['raw_checks']['single_rows']==640
scenarios=[]
for scenario in ['uniform_Scopus_20261114_P9_preserved','uniform_Scopus_20261114_P10_author_artifact']:
    for old in p9:
        c=next(r['C_new'] for r in components if r['model']==old['model'])
        p=float(old['P_scenario']) if old['P_scenario'] else None
        pool=old['P_pool']
        if scenario.endswith('P10_author_artifact') and old['model']=='Inception-1D':
            p=float(candidate['candidate']['P']);pool='author_UCR2018_128_single_vs_ensemble5_mean_error'
        u,a,f=(float(old[k]) for k in ['U_preserved','A_preserved','F_preserved'])
        low=100*(.15*c+.15*u+.25*(p if p is not None else 0)+.25*a+.20*f)
        scenarios.append({'scenario':scenario,'model':old['model'],'study_id':old['study_id'],'C':c,'U':u,'P':p,'A':a,'F':f,'P_pool':pool,'score_lower':low,'score_upper':low+(25 if p is None else 0),'known_components':5 if p is not None else 4,'complete_arithmetic_score':p is not None,'controlled_global_rank':False,'original_selection_rewritten':False,'final_test_A2_used':False})
table(B/'citation_snapshot/conditional_scores.csv',scenarios)
save(B/'citation_snapshot/scenario_protocol.json',{'date':'2026-11-14','status':'retrospective_documentary_extension_after_source_values_available','weights':{'C':.15,'U':.15,'P':.25,'A':.25,'F':.20},'C_formula':'clip(Scopus_count/(2026-earliest_study_year+1)/487,0,1)','C_snapshot':'All nine studies from the same newly supplied Scopus CSV; ten model rows because Wang is shared.','C_earliest_year_policy':'First known study release, including pinned arXiv identity; not automatically the editorial issue year. Same calendar-age and fixed487 reference policy as preserved scores.','earliest_years_by_model':{k:v[1] for k,v in source_map.items()},'P9_rule_preserved':'delivery_closure_20261114_v2/citation_score/contextual_p/preferred_intrapaper_scenario_rule.json','P10_extension':'Only Inception P is added from exact author numeric results; all five individual runs averaged equally per each of128datasets, distinct from ensemble probability averaging; source-local two-variant pool.','P10_formula':'(maxloss-singleloss)/(maxloss-minloss)','P10_is_MPCE':False,'P10_is_printed_paper_table_number':False,'cross_source_utility_equivalence_verified':False,'equal_compute_or_HPO_verified':False,'original_P_or_C_overwritten':False,'catalogue_selected_by_this_new_score':False,'missing_values':'Bounds in[0,1], never imputedzero. Equal bounds mean complete arithmetic, not a confidence interval.','global_PRISMA_closed':False,'global_optimal_Top10_selection_claimed':False,'new_model_fits':0,'coverage':[{'scenario':s,'known_components':sum(r['known_components'] for r in scenarios if r['scenario']==s),'total_components':50,'complete_scores':sum(r['complete_arithmetic_score'] for r in scenarios if r['scenario']==s)} for s in dict.fromkeys(r['scenario'] for r in scenarios)]})
print(json.dumps({'decisions':counts,'scores':load(B/'citation_snapshot/scenario_protocol.json')['coverage']},ensure_ascii=False))
