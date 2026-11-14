"""Apply the previously written retrospective P9 rule to the source-local pools."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

OUT=Path(__file__).resolve().parent


def repository_root():
    """Use an explicit repository root or locate the published script's ancestor."""
    configured = os.environ.get('E2_REPOSITORY_ROOT')
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / 'book_e2').is_dir():
            raise RuntimeError('E2_REPOSITORY_ROOT must point to a repository containing book_e2')
        return root
    for candidate in Path(__file__).resolve().parents:
        if (candidate / 'book_e2').is_dir():
            return candidate
    raise RuntimeError('Repository not found among script ancestors; set E2_REPOSITORY_ROOT')


ROOT=repository_root()
RULE=OUT/'preferred_intrapaper_scenario_rule.json'
rule=json.loads(RULE.read_text())
generated=datetime.now(timezone.utc).isoformat()
assert generated>rule['rule_fixed_at_utc']
input_path=OUT/'source_local_performance_pools.csv'
assert hashlib.sha256(input_path.read_bytes()).hexdigest()==rule['input_sha256']
pools=defaultdict(list)
for r in csv.DictReader(input_path.open()):
    pools[r['pool_id']].append(r)

def source_doi(pool_id):
    if pool_id.startswith('itransformer_'):return '10.48550/arxiv.2310.06625'
    if pool_id.startswith('timemixer_'):return '10.48550/arxiv.2405.14616'
    if pool_id.startswith('wang_'):return '10.48550/arxiv.1611.06455'
    if pool_id.startswith('sae_'):return '10.1038/s41598-019-55320-6'
    if pool_id.startswith('tsmixer_'):return '10.48550/arxiv.2303.06053'
    raise ValueError(pool_id)

forecast=['TimesNet','TiDE','PatchTST','iTransformer','TimeMixer','TSMixer']
selection_audit=[]
for model in forecast:
    eligible=[]
    for pid,rs in pools.items():
        r=rs[0]
        if not (r['metric']=='MSE' and r['dataset']=='ETTh1' and r['horizon_steps']=='96' and r['context_steps']=='96'):
            continue
        if not any(q['current_model']==model for q in rs):continue
        eligible.append((pid,len(rs),source_doi(pid)))
    eligible.sort(key=lambda q:(-q[1],q[2],int(q[0].split('_Table')[1].split('_')[0])))
    assert eligible and eligible[0][0]==rule['selected_pool_by_model'][model]
    selection_audit.append(dict(model=model,eligible_pools=json.dumps(eligible),selected_pool=eligible[0][0],
        selection_used_performance_values=False,criterion='largest comparator pool, then source DOI ascending'))

selected=[]
for model,pid in rule['selected_pool_by_model'].items():
    rs=pools[pid]
    match=[r for r in rs if r['current_model']==model]
    assert len(match)==1
    r=match[0]
    vv=[Decimal(q['value']) for q in rs]
    lo,hi=min(vv),max(vv)
    value=Decimal(r['value'])
    pvalue=(hi-value)/(hi-lo)
    selected.append(dict(model=model,P_scenario=str(pvalue),pool_id=pid,metric=r['metric'],
        comparators=' | '.join(q['source_model'] for q in rs),
        comparator_count=len(rs),source_model=r['source_model'],source_value=str(value),
        pool_min=str(lo),pool_max=str(hi),source_doi=source_doi(pid),
        source_url=r['source_url'],source_locator=r['source_locator'],
        dataset=r['dataset'],context_steps=r['context_steps'],horizon_steps=r['horizon_steps'],
        P_original='',P_contextual_global='',scenario_id=rule['scenario_id'],
        posthoc=True,controlled_global_ranking=False,final_test_A2_used=False))

def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

for r in selected:r['availability_reason']='selected_by_retrospective_rule'
assert len(selected)==9 and sum(r['P_scenario']!='' for r in selected)==9
assert all(0<=Decimal(r['P_scenario'])<=1 for r in selected if r['P_scenario']!='')
assert all(r['P_original']==r['P_contextual_global']=='' for r in selected)
save('preferred_intrapaper_scenario.csv',selected)
save('preferred_intrapaper_selection_audit.csv',selection_audit)
sae=[]
for metric in ['RMSE','MAE','SMAPE']:
    rs=pools['sae_Beijing_Table2_4_'+metric]
    r=next(q for q in rs if q['current_model']=='LSTM-autoencoder con predictor')
    values=[Decimal(q['value']) for q in rs]
    sae.append(dict(model=r['current_model'],metric=metric,source_value=r['value'],
        P_scenario_metric_sensitivity=str((max(values)-Decimal(r['value']))/(max(values)-min(values))),
        selected_for_P9=metric=='RMSE',pool_id=r['pool_id'],note='No metric average; RMSE chosen by the fixed retrospective scenario rule.'))
save('preferred_intrapaper_SAE_metric_sensitivity.csv',sae)

receipt=dict(scenario_id=rule['scenario_id'],rule_fixed_at_utc=rule['rule_fixed_at_utc'],
    scenario_generated_at_utc=generated,rule_sha256=hashlib.sha256(RULE.read_bytes()).hexdigest(),
    scenario_sha256=hashlib.sha256((OUT/'preferred_intrapaper_scenario.csv').read_bytes()).hexdigest(),
    rule_written_before_selection_generation=True,preregistration_claimed=False,
    previous_metric_values_already_available=True,selected_numeric_rows=9,current_model_rows=len(selected),
    forecast_rule_automatically_checked=True,source_pool_integrity_checked=True,
    P_original_written=False,P_contextual_global_written=False,final_test_A2_read=False)
(OUT/'preferred_intrapaper_scenario_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps([{'model':r['model'],'P_scenario':r['P_scenario'],'pool_id':r['pool_id']} for r in selected],indent=2))
