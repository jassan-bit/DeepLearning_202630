#!/usr/bin/env python3
"""Reanalyze recorded HPO descriptors using only Python's standard library.

Extract the standalone ZIP and run: python replay_descriptive.py
No models, market data, downloads, packages, fitting or statistical testing.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
PARAMS = ('learning_rate','weight_decay','batch_size','dropout','warmup_fraction')


def rows(name):
    with (ROOT/'inputs'/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    for row in manifest['files']:
        p=ROOT/row['path']
        if p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Input changed: '+row['path'])
    configs=rows('configurations_78.csv');opportunities=rows('opportunities_234.csv');charges=rows('charges_234.csv')
    expected=rows('marginal_associations_expected.csv')
    lookup={(r['model'],r['parameter'],float(r['value'])):r for r in expected}
    models=list(dict.fromkeys(r['model'] for r in configs));differences=[];compared=0;one_factor_pairs=0
    for model in models:
        group=[r for r in configs if r['model']==model]
        for a,b in itertools.combinations(group,2):
            changed=sum(float(a[p] or 0)!=float(b[p] or 0) for p in PARAMS)
            one_factor_pairs+=changed==1
        for p in PARAMS:
            for level in sorted({float(r[p] or 0) for r in group}):
                chosen=[float(r['pooled_cv_RMSE']) for r in group
                        if float(r[p] or 0)==level and r['eligible_for_selection']=='True']
                reference=lookup[(model,p,level)]
                if not chosen:
                    assert reference['pooled_cv_RMSE_mean_descriptive']==''
                    continue
                for key,value in [('pooled_cv_RMSE_mean_descriptive',statistics.mean(chosen)),
                                  ('pooled_cv_RMSE_min_descriptive',min(chosen)),
                                  ('pooled_cv_RMSE_max_descriptive',max(chosen))]:
                    differences.append(abs(value-float(reference[key])));compared+=1
    observed=[float(r['observed_wall_charge_seconds']) for r in charges if r['observed_wall_charge_seconds']]
    unknown=[float(r['unmeasured_conservative_allocation_seconds']) for r in charges if r['unmeasured_conservative_allocation_seconds']]
    result={'models':len(models),'configurations':len(configs),'opportunities':len(opportunities),
            'completed_fold_outcomes':sum(r['status']=='completed' for r in opportunities),
            'recorded_winners':sum(r['recorded_winner']=='True' for r in configs),
            'one_factor_only_comparison_pairs':one_factor_pairs,
            'observed_charge_receipts':len(observed),'observed_charge_seconds_mixed_scopes':sum(observed),
            'unknown_allocation_seconds_separate_not_added':sum(unknown),
            'marginal_numeric_values_reconstructed':compared,'maximum_marginal_absolute_difference':max(differences),
            'new_fits':0,'model_forward_passes':0,'network_calls':0,'resampling_draws':0,
            'independent_importance_or_selection_variance_established':False,
            'native_original_coverage_recovered_by_this_replay':False}
    assert (len(models),len(configs),len(opportunities),result['completed_fold_outcomes'],len(observed),one_factor_pairs)==(13,78,234,231,233,1)
    assert max(differences)==0
    assert abs(sum(observed)-25833.89891848399)<1e-8
    assert sum(unknown)==1860
    result['status']='passed'
    (ROOT/'replay_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
