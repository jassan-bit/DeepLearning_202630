#!/usr/bin/env python3
"""Compute explicitly conditional posthoc sensitivity; never modify original P."""
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil

OUT = pathlib.Path(__file__).resolve().parent


def repository_root():
    """Use an explicit repository root or locate the published script's ancestor."""
    configured = os.environ.get('E2_REPOSITORY_ROOT')
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if not (root / 'book_e2').is_dir():
            raise RuntimeError('E2_REPOSITORY_ROOT must point to a repository containing book_e2')
        return root
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / 'book_e2').is_dir():
            return candidate
    raise RuntimeError('Repository not found among script ancestors; set E2_REPOSITORY_ROOT')


ROOT = repository_root()
OLD = ROOT/'references/literature_e2/final_audit_20261113'
WEIGHTS = {'C':.15,'U':.15,'P':.25,'A':.25,'F':.20}
AUDIT = OLD/'current10_score_component_audit.csv'
PREFERRED = OUT/'contextual_p/preferred_intrapaper_scenario.csv'
P_RULE = OUT/'contextual_p/preferred_intrapaper_scenario_rule.json'
P_RECEIPT = OUT/'contextual_p/preferred_intrapaper_scenario_receipt.json'
P_POOLS = OUT/'contextual_p/source_local_performance_pools.csv'

def read(path):
    return list(csv.DictReader(path.open()))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_label(path):
    """Keep provenance portable for both repository inputs and local artifacts."""
    if path.is_relative_to(OUT):
        return 'artifact:' + str(path.relative_to(OUT))
    return 'repository:' + str(path.relative_to(ROOT))

base=read(AUDIT)
preferred={r['model']:r for r in read(PREFERRED)}
rule=json.loads(P_RULE.read_text())
receipt=json.loads(P_RECEIPT.read_text())
assert receipt['rule_sha256']==sha(P_RULE), 'P9 rule does not match its generation receipt'
assert receipt['scenario_sha256']==sha(PREFERRED), 'P9 CSV does not match its generation receipt'
assert rule['input_sha256']==sha(P_POOLS), 'P9 source pools do not match the fixed rule'
assert len(base)==10 and len(preferred)==9
assert {r['model'] for r in base}-set(preferred)=={'Inception-1D'}
assert all(r['P']=='' for r in base)
old_sensitivity=read(OLD/'posthoc_contextual_score_sensitivity.csv')
shutil.copyfile(OLD/'posthoc_contextual_score_sensitivity.csv',OUT/'prior_contextual_sensitivity_retained.csv')
assert sha(OLD/'posthoc_contextual_score_sensitivity.csv')==sha(OUT/'prior_contextual_sensitivity_retained.csv')
rows=[]

def add_scenario(scenario, p_by_model, p_pools, metric_policy, provenance, cross_source_calibration_assumed):
    for old in base:
        m=old['model']
        values={k:(float(old[k]) if old[k]!='' else None) for k in WEIGHTS}
        values['P']=p_by_model.get(m)
        known=sum(WEIGHTS[k] for k,v in values.items() if v is not None)
        lower=100*sum(WEIGHTS[k]*v for k,v in values.items() if v is not None)
        upper=lower+100*(1-known)
        rows.append(dict(scenario=scenario,model=m,study_id=old['study_id'],
            C_preserved=values['C'],U_preserved=values['U'],P_original=old['P'],
            P_scenario=values['P'],A_preserved=values['A'],F_preserved=values['F'],
            P_pool=p_pools.get(m,''),metric_policy=metric_policy,
            conditional_score_lower=round(lower,8),conditional_score_upper=round(upper,8),
            known_component_count=sum(v is not None for v in values.values()),
            known_weight=round(known,8),unknown_weight=round(1-known,8),
            complete_under_this_scenario=all(v is not None for v in values.values()),
            cross_source_calibration_assumed=cross_source_calibration_assumed,
            cross_source_calibration_verified=False,controlled_global_ranking=False,
            global_definitive_score=False,original_P_overwritten=False,
            original_catalogue_membership_changed=False,final_test_A2_used=False,
            provenance=provenance))

add_scenario('original_component_audit_preserved',{}, {},'original P remains unidentified',str(AUDIT.relative_to(ROOT)),False)
for scenario in sorted({r['scenario'] for r in old_sensitivity}):
    subset=[r for r in old_sensitivity if r['scenario']==scenario]
    add_scenario(scenario,{r['model']:float(r['P_contextual']) for r in subset if r['P_contextual']!=''},
                 {r['model']:r['stratum'] for r in subset},
                 'prior sensitivity retained; nominal source MSE across papers',
                 'prior_contextual_sensitivity_retained.csv',True)
newp={m:float(r['P_scenario']) for m,r in preferred.items() if r['P_scenario']!=''}
newpools={m:r['pool_id'] for m,r in preferred.items()}
add_scenario('posthoc_intrapaper_9_primary',newp,newpools,
             'MPCE classification; RMSE Beijing; MSE ETTh1; pools fixed by rule before scenario scoring',
             'contextual_p/preferred_intrapaper_scenario.csv',True)

# Counterfactual metric sensitivity, never a best-metric selection or average.
candidates=read(OUT/'contextual_p/current_model_contextual_P_candidates.csv')
for metric in ['MAE','SMAPE']:
    alternate=newp.copy()
    match=[r for r in candidates if r['current_model']=='LSTM-autoencoder con predictor' and r['metric']==metric]
    assert len(match)==1
    alternate['LSTM-autoencoder con predictor']=float(match[0]['P_contextual_source_local'])
    pools=newpools.copy();pools['LSTM-autoencoder con predictor']=match[0]['pool_id']
    add_scenario('posthoc_intrapaper_9_SAE_'+metric+'_sensitivity',alternate,pools,
                 'Primary scenario with SAE '+metric+' only; metric sensitivity, not selected winner',
                 'contextual_p/current_model_contextual_P_candidates.csv',True)

with (OUT/'conditional_score_by_scenario.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary=[]
for scenario in dict.fromkeys(r['scenario'] for r in rows):
    subset=[r for r in rows if r['scenario']==scenario]
    summary.append(dict(scenario=scenario,known_C=sum(r['C_preserved'] is not None for r in subset),
        known_U=10,known_P=sum(r['P_scenario'] is not None for r in subset),known_A=10,known_F=10,
        known_components=sum(r['known_component_count'] for r in subset),total_components=50,
        complete_conditional_scores=sum(r['complete_under_this_scenario'] for r in subset),models=10,
        definitive_global_score=False,original_scores_replaced=False))
with (OUT/'component_coverage_by_scenario.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
primary=next(r for r in summary if r['scenario']=='posthoc_intrapaper_9_primary')
assert primary['known_components']==45 and primary['complete_conditional_scores']==6
assert primary['known_C']==6 and primary['known_P']==9
protocol={
    'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
    'status':'new retrospective conditional sensitivity; not original selection, preregistration, or controlled global rank',
    'weights':WEIGHTS,'formula':'100*(.15C+.15U+.25P_scenario+.25A+.20F)',
    'missing_rule':'Algebraic bounds vary nulls in[0,1]; absent never zero; bounds are not confidence intervals.',
    'preferred_P_rule':'contextual_p/preferred_intrapaper_scenario_rule.json',
    'primary_scenario':'posthoc_intrapaper_9_primary',
    'C_provider_and_snapshot':'All six C unchanged from the same received Scopus export; four null. No publisher counts imported.',
    'cross_source_calibration':'The arithmetic sensitivity assumes local0-1scales are utility-equivalent across source pools; this calibration is not established by the papers. Therefore no definitive global score or ranking is claimed.',
    'interpretation':'New9model scenario improves documented local comparisons versus prior6model nominal ETTh1 sensitivity. Tables strengthen within-source comparison; cross-source equivalence, equal HPO and BTC transfer remain unresolved.',
    'original_P_unchanged':True,'original_membership_unchanged':True,'final_test_A2_used':False,
    'P9_rule_receipt_and_pool_integrity_checked':True,
    'input_sha256':{input_label(p):sha(p) for p in [AUDIT,PREFERRED,P_RULE,P_RECEIPT,P_POOLS,OLD/'posthoc_contextual_score_sensitivity.csv',OUT/'contextual_p/current_model_contextual_P_candidates.csv']},
    'coverage':summary,
}
(OUT/'conditional_score_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
