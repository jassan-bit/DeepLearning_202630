#!/usr/bin/env python3
"""Compute source-stratified descriptive scores and robust unidentified-P bounds."""
import csv
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCE = REPO / 'references/literature_e2/delivery_closure_20261114_v2/citation_score/contextual_p/source_local_performance_pools.csv'
SCORES = REPO / 'book_e2/_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/conditional_scores.csv'
EXTRACT = REPO / 'book_e2/_static/e2_literature_closure_20261113/current10_extraction.csv'


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = read(SOURCE)
    scores = {r['model']: r for r in read(SCORES) if r['scenario'] == 'uniform_Scopus_20261114_P10_author_artifact'}
    extraction = {r['model']: r for r in read(EXTRACT)}
    assert len(scores) == len(extraction) == 10
    # Non-performance evidence; no missing P is assigned a point value.
    base = {m: 100*(.15*float(r['C'])+.15*float(r['U'])+.25*float(r['A'])+.20*float(r['F'])) for m,r in scores.items()}
    pools = collections.defaultdict(list)
    for row in source:
        row = dict(row)
        # The source receipt explicitly gives SAE-2 lag25 and all other rows lag30.
        if row['dataset'] == 'Beijing PM2.5':
            row['context_steps'] = '25' if row['source_model'] == 'LSTM-SAE-2' else '30'
        key = (row['pool_id'], row['task_family'], row['dataset'], row['task'], row['context_steps'], row['horizon_steps'], row['metric'])
        pools[key].append(row)
    # Author artifacts are a separate stratum, not MPCE/UCR44 or paper UCR85.
    candidate = json.loads((REPO / 'book_e2/_static/e2_literature_delivery_closure_20261114_v3/inception_single/candidate.json').read_text())['candidate']
    for label, model, value in [('single_network', 'Inception-1D', candidate['single_mean_error']), ('ensemble5', '', candidate['ensemble5_mean_error'])]:
        row = {'pool_id':'author_UCR128_two_variants','task_family':'classification','dataset':'UCR2018/128','task':'mean classification error over all128datasets','context_steps':'archive-provided variable sequence lengths','horizon_steps':'NA','metric':'mean_classification_error','source_model':label,'current_model':model,'value':str(value),'source_url':'author numeric artifacts','source_locator':'results-inception-128.csv and results-InceptionTime-128.csv','caveat':'Same named dataset set; exact split bytes and equal compute unverified. No cross-pool calibration.'}
        pools[tuple(row[k] for k in ('pool_id','task_family','dataset','task','context_steps','horizon_steps','metric'))].append(row)
    stratified, coverage = [], []
    for key, rows in pools.items():
        losses = [float(r['value']) for r in rows]
        lo, hi = min(losses), max(losses)
        valid = len(rows)>1 and hi>lo
        local = {r['current_model']:base[r['current_model']]+25*(hi-float(r['value']))/(hi-lo) for r in rows if r['current_model'] and valid}
        current = sorted({r['current_model'] for r in rows if r['current_model']})
        coverage.append({'stratum_id':key[0]+'__L'+key[4], 'task':key[3], 'dataset':key[2], 'lookback':key[4], 'horizon':key[5], 'metric':key[6], 'source_candidates':len(rows), 'current_model_count':len(current), 'current_models':' | '.join(current), 'families_current':' | '.join(sorted({extraction[m]['family'] for m in current})), 'same_source_pool_only':True})
        for r in rows:
            if not r['current_model']:
                continue
            m = r['current_model']
            p = (hi-float(r['value']))/(hi-lo) if valid else None
            stratified.append({'model':m,'stratum_id':key[0]+'__L'+key[4],'task':key[3],'dataset':key[2],'lookback':key[4],'horizon':key[5],'metric':key[6],'source_loss':r['value'],'stratum_candidate_count':len(rows),'normalization_min':lo,'normalization_max':hi,'P_within_stratum':p if p is not None else '', 'S_within_source_stratum':local.get(m,''),'rank_among_current_same_stratum':1+sum(v>local[m]+1e-12 for v in local.values()) if m in local else '', 'current_comparison_count':len(current),'P_unidentified_reason':'' if valid else 'Singleton after exact-lookback stratification; no range normalization','source_locator':r['source_locator'],'source_url':r['source_url'],'interpretation':'Descriptive source-table priority; no cross-stratum utility or controlled significance claim'})
    write('stratified_scores.csv',stratified)
    write('stratum_coverage.csv',coverage)
    bounds, dominance = [], []
    for wp in (.10,.25,.40):
        intervals = {m:((1-wp)*b/.75, (1-wp)*b/.75+100*wp) for m,b in base.items()}
        for m,(lo,hi) in intervals.items():
            higher = [j for j,(jl,ju) in intervals.items() if j!=m and jl>hi+1e-12]
            lower = [j for j,(jl,ju) in intervals.items() if j!=m and lo>ju+1e-12]
            bounds.append({'model':m,'P_weight':wp,'C':scores[m]['C'],'U':scores[m]['U'],'A':scores[m]['A'],'F':scores[m]['F'],'P_global':'unidentified; independently bounded[0,1]','score_lower':lo,'score_upper':hi,'best_rank':1+len(higher),'worst_rank':10-len(lower),'guaranteed_above':' | '.join(lower),'guaranteed_below':' | '.join(higher),'nonP_component_ratios_preserved':True,'historical_weights_changed':False,'scenario':'principal' if wp==.25 else 'declared_sensitivity_only'})
            for j in lower:
                dominance.append({'P_weight':wp,'dominant_model':m,'dominated_model':j,'minimum_score_advantage':lo-intervals[j][1],'meaning':'For all independently allowed globalP; documentary score dominance, not predictive superiority'})
    write('robust_global_P_bounds.csv',bounds)
    write('robust_dominance.csv',dominance)
    reasons = {
        'FCN-1D':'Convolución temporal simple y pooling global; transferencia de clasificación a regresión declarada.',
        'ResNet-1D':'Conexiones residuales sobre convoluciones; permite contrastar profundidad/optimización frente a FCN con fuente compartida.',
        'Inception-1D':'Ramas convolucionales multiescala; cuerpo individual separado del ensemble y kernels adaptados a D7.',
        'LSTM-autoencoder con predictor':'Representación recurrente mediante reconstrucción y transferencia a predictor; aporta familia distinta y preentrenamiento explícito.',
        'TimesNet':'Periodización y convoluciones2D: añade una hipótesis estructural a convolución temporal directa.',
        'TSMixer':'Mezcla temporal y entre variables mediante MLP; representación sin atención.',
        'TiDE':'Encoder-decoder denso residual y covariables; conserva cuerpo publicado y cabeza adaptada.',
        'TimeMixer':'Descomposición y mezcla multiescala con MLP; hipótesis diferente de mezcla plana.',
        'PatchTST':'Atención sobre parches temporales; adaptación explícita del tamaño/stride a D7.',
        'iTransformer':'Atención sobre tokens de variables; contrasta eje de atención con parches temporales.',
    }
    write('catalogue_selection_rationale.csv',[{'model':m,'family':extraction[m]['family'],'study_id':extraction[m]['study_id'],'criterion':'Representatividad de mecanismo + código identificado + transferencia D7 explícita; desempeño sólo en estratos comparables','rationale':reasons[m],'global_rank_claimed':False,'prior_score_selection_claimed':False,'source_pointer':'current10_extraction.csv#model='+m} for m in scores])
    exclusions = read(REPO/'references/literature_e2/final_audit_20261113/excluded_named_originals_scope_audit.csv')
    write('named_exclusions_preserved.csv',exclusions)
    main_bounds = [r for r in bounds if r['P_weight']==.25]
    assert all(r['best_rank']==1 and r['worst_rank']==10 for r in main_bounds)
    assert not any(r['P_weight']==.25 for r in dominance)
    print('Principal wP=.25: all10rank intervals1–10;0robust dominance pairs.')
    print('Sensitivity dominance pairs:',dict(collections.Counter(r['P_weight'] for r in dominance)))
    print('Local current-model counts:',[(r['stratum_id'],r['current_model_count'],r['families_current']) for r in coverage if r['dataset']=='ETTh1' and r['metric']=='MSE'])
    print('Principal bounds:',[(r['model'],round(r['score_lower'],2),round(r['score_upper'],2)) for r in main_bounds])


if __name__=='__main__':
    main()
