"""Build post-hoc, source-local performance descriptions; never alter selection P."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent


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


ROOT = repository_root()
OBSERVED = datetime.now(timezone.utc).isoformat()
CURRENT = {
    'FCN': 'FCN-1D', 'ResNet': 'ResNet-1D', 'InceptionTime(5)': 'Inception-1D',
    'LSTM-SAE-2': 'LSTM-autoencoder con predictor', 'TimesNet': 'TimesNet',
    'TSMixer': 'TSMixer', 'TiDE': 'TiDE', 'TimeMixer': 'TimeMixer',
    'PatchTST': 'PatchTST', 'iTransformer': 'iTransformer',
}

def table(pool_id, family, dataset, task, metric, values, source_url, source_version,
          locator, context='', horizon='', evidence='new_official_HTML_table_read', caveat=''):
    minimum, maximum = min(values.values()), max(values.values())
    assert maximum > minimum
    return [dict(pool_id=pool_id, task_family=family, dataset=dataset, task=task,
        metric=metric, lower_is_better=True, source_model=k,
        current_model=CURRENT.get(k, ''), context_steps=context, horizon_steps=horizon,
        value=v, pool_n=len(values), pool_min=minimum, pool_max=maximum,
        P_contextual_source_local=(maximum-v)/(maximum-minimum),
        P_original_overwritten=False, cross_pool_comparable=False,
        original_selection_cause=False, final_test_A2_used=False,
        source_url=source_url, source_version=source_version, source_locator=locator,
        evidence=evidence, caveat=caveat) for k,v in values.items()]

rows=[]
rows += table('wang_UCR44_TableI_MPCE_11', 'classification', 'UCR44',
    'univariate series classification; aggregate over 44 datasets', 'MPCE',
    dict(zip(['DTW','COTE','MCNN','BOSSVS','PROP','BOSS','SE1','TSBF','MLP','FCN','ResNet'],
    [.0397,.0226,.0241,.0330,.0304,.0256,.0302,.0335,.0407,.0219,.0231])),
    'https://arxiv.org/html/1611.06455v4', '1611.06455v4',
    'Table I, MPCE row; Section III-B Eq.4; III-C',
    caveat='All 11 reported candidates, not only the 2 current models; normalized published MPCE, not a published P. No equal HPO or seed-uncertainty claim; paired tests do not establish an FCN/ResNet difference.')

sae_values={
    'RMSE': {'DLSTM-1':23.993,'DLSTM-2':23.750,'LSTM-SAE-1':23.907,'LSTM-SAE-2':24.041},
    'MAE': {'DLSTM-1':12.124,'DLSTM-2':12.452,'LSTM-SAE-1':12.509,'LSTM-SAE-2':12.060},
    'SMAPE': {'DLSTM-1':10.919,'DLSTM-2':12.181,'LSTM-SAE-1':11.052,'LSTM-SAE-2':9.864},
}
for metric,values in sae_values.items():
    rows += table('sae_Beijing_Table2_4_'+metric, 'forecasting', 'Beijing PM2.5',
        'hourly next-step scalar forecasting', metric, values,
        'https://pmc.ncbi.nlm.nih.gov/articles/PMC6911101/',
        'Scientific Reports 2019; PMCID PMC6911101', 'Table 2',
        context='25 for LSTM-SAE-2; 30 for other rows', horizon=1,
        caveat='Source comparison permits different selected lags/hidden units. Chronology and equal HPO budgets unverified. Three metrics are separate sensitivity views; no average P is selected.')

models=json.loads((ROOT/'references/literature_e2/fulltext_20261105/models.json').read_text())['models']
im=next(m for m in models if m['model']=='iTransformer')
fullrow=im['extraction']['exact_metric_pair']['full_row']
for metric in ('MSE','MAE'):
    rows += table('itransformer_ETTh1_H96_L96_Table10_11_'+metric,
        'forecasting','ETTh1','multivariate forecast of all seven variables',metric,
        {r['model']:r[metric] for r in fullrow},
        'https://arxiv.org/abs/2310.06625v4','2310.06625v4',
        'Table 10, ETTh1, H96 row; PDF p23', context=96,horizon=96,
        evidence='existing_full_PDF_extraction_with_visual_verified_full_row',
        caveat='All 11 models in this source protocol. Current architectures are contextual only; TiDE and PatchTST values differ from their original-paper long-context values. Equal HPO effort and seed uncertainty not harmonized.')

tm_names=['TimeMixer','PatchTST','TimesNet','Crossformer','MICN','FiLM','DLinear','FEDformer','Stationary','Autoformer','Informer']
tm_mse=[.375,.460,.384,.423,.426,.438,.397,.395,.513,.449,.865]
tm_mae=[.400,.447,.402,.448,.446,.433,.412,.424,.491,.459,.713]
for metric,values in [('MSE',tm_mse),('MAE',tm_mae)]:
    rows += table('timemixer_ETTh1_H96_L96_Table13_11_'+metric,
        'forecasting','ETTh1','multivariate forecast of all seven variables',metric,
        dict(zip(tm_names,values)), 'https://arxiv.org/html/2405.14616v1',
        '2405.14616v1','Appendix E, Table 13, ETTh1 H96 row',context=96,horizon=96,
        caveat='Unified input lengths, batch sizes and epochs reported by source; global metric scale remains unverified. Different from searched Table14; no equal total HPO-cost claim.')
for metric,values in [('MSE',[.375,.401,.418,.387]),('MAE',[.400,.428,.437,.411])]:
    rows += table('timemixer_ETTh1_H96_L96_Table23_4_'+metric,
        'forecasting','ETTh1','multivariate forecast of all seven variables',metric,
        dict(zip(['TimeMixer','Scaleformer','MTSMixer','TSMixer'],values)),
        'https://arxiv.org/html/2405.14616v1','2405.14616v1',
        'Appendix G, Table 23, ETTh1 H96 row, Unified Hyperparameter columns',context=96,horizon=96,
        caveat='TSMixer is the TimeMixer authors\' PyTorch reproduction, not the TSMixer original result. Table23 is a separate baseline pool; no merging with Table13 or searched columns.')

# The paper explicitly skips TMix-Only in comparisons; use seven comparison candidates.
ts_names=['TSMixer','TFT','FEDformer','Autoformer','Informer','Linear','PatchTST']
for metric,values in [('MSE',[.361,.674,.376,.435,.941,.368,.370]),('MAE',[.392,.634,.415,.446,.769,.392,.400])]:
    rows += table('tsmixer_ETTh1_H96_L512_Table3_7_'+metric,
        'forecasting','ETTh1','multivariate forecast of all seven variables',metric,
        dict(zip(ts_names,values)), 'https://arxiv.org/html/2303.06053v5',
        '2303.06053v5','Table 3, ETTh1 H96 row; exclude TMix-Only as caption instructs',context=512,horizon=96,
        caveat='Starred comparator scores copied from PatchTST, not all rerun with equal budgets. L512 describes TSMixer and PatchTST; other baselines retain source protocol. Source-local comparison, not controlled benchmark.')

with (OUT/'source_local_performance_pools.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
current_rows=[r for r in rows if r['current_model']]
with (OUT/'current_model_contextual_P_candidates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(current_rows)

source_audit=list(csv.DictReader((ROOT/'references/literature_e2/final_audit_20261113/current10_score_component_audit.csv').open()))
summary=[]
for old in source_audit:
    candidates=[r for r in current_rows if r['current_model']==old['model']]
    summary.append(dict(model=old['model'],study_id=old['study_id'],P_original=old['P'],
        contextual_candidate_rows=len(candidates),
        contextual_pool_ids=' | '.join(r['pool_id'] for r in candidates),
        P_contextual_global='',global_total_score_computed=False,
        status='source_local_candidates_available' if candidates else 'single_network_metric_not_verified',
        selection_original_unchanged=True,
        unresolved='Cross-pool calibration, metric/pool choice, protocol harmonization; do not fill original P' if candidates else '54/8/23 compares five-network ensembles; current single network cannot inherit this score'))
with (OUT/'current10_P_adjudication_v2.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

inception=dict(source_url='https://arxiv.org/html/1909.04939v3',source_version='1909.04939v3',
    locator='Section 5, Figure 10; Figure 11; Section 3.2',source_variant='InceptionTime(5)',
    comparator='ResNet(5), batch64',datasets=85,wins=54,ties=8,losses=23,
    descriptive_win_credit=(54+8*.5)/85,formula='(wins+0.5*ties)/datasets',
    current_model='Inception-1D',P_contextual_current_single_network=None,
    reason='Reported ensemble comparison does not identify single-network current-model performance.',
    published_p_value='<0.01',claim='architecture-context fact, not a P score for the implemented single network')
(OUT/'inception_ensemble_boundary.json').write_text(json.dumps(inception,ensure_ascii=False,indent=2)+'\n')

inputs=[ROOT/'references/literature_e2/fulltext_20261105/models.json',
    ROOT/'references/literature_e2/final_audit_20261113/current10_score_component_audit.csv',
    ROOT/'references/literature_e2/final_audit_20261113/posthoc_contextual_score_protocol.json',
    ROOT/'references/literature_e2/final_audit_20261113/contextual_performance_comparability.csv']
protocol=dict(version='e2.P_source_local_posthoc.v2',created_at_utc=OBSERVED,
    status='posthoc_descriptive_proposal_not_original_selection_protocol',
    original_P_unchanged=True,original_catalogue_membership_unchanged=True,
    original_score_unchanged=True,final_test_A2_used=False,
    formula='P_contextual_source_local=(max_pool_metric-metric)/(max_pool_metric-min_pool_metric)',
    domains='Each pool fixes source/table/dataset/task/metric and the listed comparison set; do not compare across pool IDs.',
    sigma7_identity_required=False,
    interpretation='Comparable external datasets can inform architectural literature context. Neither BTC effectiveness nor a universal Top10 rank follows.',
    pooling_rules=['Separate classification and forecasting','Separate RMSE, MAE, SMAPE, MPCE and MSE',
        'Separate original results from another source\'s reproduction','Do not merge different lookbacks, source versions, selected/unaltered settings or baseline pools',
        'A source-local zero means observed worst in its pool; an absent value remains null'],
    source_review_scope='New reads of the identified official HTML tables and captions only; existing iTransformer PDF full-row extraction reused. No new complete-paper or human review claimed.',
    covered_current_models=sorted({r['current_model'] for r in current_rows}),
    current_models_with_contextual_candidates=9,source_pool_metric_rows=len(rows),current_candidate_rows=len(current_rows),
    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
(OUT/'protocol_and_provenance.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
assert len(summary)==10 and sum(bool(s['contextual_candidate_rows']) for s in summary)==9
assert all(s['P_original']=='' and s['P_contextual_global']=='' for s in summary)
assert all(0<=r['P_contextual_source_local']<=1 for r in rows)
assert all(not r['P_original_overwritten'] and not r['original_selection_cause'] and not r['final_test_A2_used'] for r in rows)
print(json.dumps({k:protocol[k] for k in ('current_models_with_contextual_candidates','source_pool_metric_rows','current_candidate_rows')},indent=2))
