#!/usr/bin/env python3
"""Join published E2 benchmark tables without fitting, forecasting or testing.

Only the standard library is used. Inputs are hash-pinned. This program copies
published estimates and their scopes; ordinal display order is not a new test.
Run with --output-dir pointing to a new directory. Existing outputs are refused.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

SOURCES = {
    "a2": {
        "path": "book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv",
        "sha256": "879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1"
    },
    "analysis": {
        "path": "book_e2/_static/e2_benchmark_final_20261113/benchmark_analysis.json",
        "sha256": "73206e1aa5bf004b7e0f884ba3685c0bdfbed209425c6397a68fa280c04ff637"
    },
    "efficiency": {
        "path": "book_e2/_static/e2_efficiency_closure_20261113/model_efficiency.csv",
        "sha256": "973e54a8c56ca3fd113accd26413d5322bb4b72dbcc5caa73950a77c52129567"
    },
    "latency": {
        "path": "book_e2/_static/e2_posthoc_gpu_20261113/inference_profiles_by_model.csv",
        "sha256": "9a517bdc60282fa6d2362477f2ec5799451cd0b74d96c88ab188b2ff66a1fd1f"
    },
    "flops": {
        "path": "book_e2/_static/e2_flops_final_audit_20261114/flops_by_model.csv",
        "sha256": "dc87d3c6d2e9d880fe62fcfd6448abb4c4126ff106accf49bd2875e4547c5dc0"
    },
    "flops_pareto": {
        "path": "book_e2/_static/e2_flops_final_audit_20261114/pareto_core_flops_rmse.csv",
        "sha256": "d5d99d3c72354f568fd923ab946a5f9937ea6eb69e18b194240b670605e49c9c"
    },
    "robustness": {
        "path": "book_e2/_static/e2_posthoc_gpu_20261113/normalized_degradation_areas.csv",
        "sha256": "871000659803d5dcd83b04ccf442cddbc6e914d75e0caf52eb6638f89d2a4825"
    },
    "ranks": {
        "path": "book_e2/_static/e2_f2a_s8_20261113/conditional_seed_ranks.csv",
        "sha256": "40229ce75e8449c2829740efbd3d62b70f1900c9cbb48f42ce4fdcba34612c4b"
    },
    "mcs": {
        "path": "book_e2/_static/e2_f2a_s8_20261113/mcs_membership.csv",
        "sha256": "38f118090eb50c1ecf357c3078588eaa12befee4f9c26aaae255260a8eceed54"
    },
    "calibration": {
        "path": "book_e2/_static/e2_conditional_calibration_20261114/model_metrics.csv",
        "sha256": "463bfe0466764330cf35d91cd44f0ca637684cf1ce00e83c73286aac8e0b1d22"
    }
}


def rows(root, key):
    return list(csv.DictReader(io.StringIO((root / SOURCES[key]['path']).read_text())))


def indexed(data, key='model'):
    result = {r[key]: r for r in data}
    if len(result) != len(data):
        raise ValueError('Duplicate source join key')
    return result


def assemble(root):
    for source in SOURCES.values():
        if hashlib.sha256((root/source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Input hash mismatch: '+source['path'])
    a2 = rows(root, 'a2')
    analysis = indexed(json.loads((root/SOURCES['analysis']['path']).read_text())['table_A2'])
    tables = {key:indexed(rows(root,key)) for key in ['efficiency','latency','flops','flops_pareto','ranks','mcs']}
    areas = {(r['model'],r['kind']):r for r in rows(root,'robustness')}
    if len(areas) != 52:
        raise ValueError('Expected 13 models by four stressors')
    calibration = [r for r in rows(root,'calibration') if r['policy']=='original_frozen' and r['group']=='global']
    calibration = indexed(calibration)
    models = {r['model'] for r in a2}
    if len(models)!=16 or set(analysis)!=models or set(tables['efficiency'])!=models or set(calibration)!=models:
        raise ValueError('Expected complete sixteen-model source support')
    results=[]
    for position, a in enumerate(sorted(a2,key=lambda r:float(r['RMSE'])),1):
        model=a['model']; original=analysis[model]; e=tables['efficiency'][model]
        if float(a['RMSE']) != original['RMSE'] or a['RMSE']!=e['RMSE']:
            raise ValueError('Frozen RMSE identity mismatch')
        for key in ['RMSE_seed_SD','MAE','sMAPE','R2_mean','coverage90','mean_interval_width90']:
            if a[key] and abs(float(a[key])-original[key])>1e-15:
                raise ValueError('A2 source mismatch: '+key)
        ci=original['RMSE_seed_t_CI95']
        if int(a['R'])==1 and (ci is not None or a['RMSE_seed_SD']):
            raise ValueError('Deterministic predictors cannot have invented seed uncertainty')
        rank=tables['ranks'].get(model,{}); f=tables['flops'].get(model,{})
        lat=tables['latency'].get(model,{}); mcs=tables['mcs'][model]
        if f and f['registered_parameters'] != e['actual_parameter_count']:
            raise ValueError('Parameter source mismatch')
        if rank and abs(float(rank['mean_seed_RMSE'])-float(a['RMSE']))>1e-15:
            raise ValueError('Conditional rank metric mismatch')
        cal=calibration[model]
        if abs(float(cal['coverage'])-float(a['coverage90']))>1e-14:
            raise ValueError('Original calibration scope mismatch')
        r=dict(a)
        r.update({
            'display_position_by_RMSE':position,
            'RMSE_seed_t_CI95_lower':ci[0] if ci else None,
            'RMSE_seed_t_CI95_upper':ci[1] if ci else None,
            'seed_CI_scope':original['seed_CI_scope'],
            'conditional_neural_mean_rank':rank.get('mean_rank'),
            'conditional_neural_min_rank':rank.get('min_rank'),
            'conditional_neural_max_rank':rank.get('max_rank'),
            'actual_parameter_count':e['actual_parameter_count'],
            'final_phase_seconds_mean_per_seed':e['final_phase_seconds_mean_per_seed'],
            'final_phase_seconds_SD_across_seeds':e['final_phase_seconds_SD_across_seeds'],
            'final_phase_seconds_sum':e['final_phase_seconds_sum'],
            'final_training_peak_allocated_bytes_observed':e['final_training_peak_allocated_bytes_observed'],
            'final_memory_scope':e['final_memory_scope'],
            'full_selection_seconds':e['full_selection_seconds'],
            'mean_latency_gpu_median_ms':lat.get('mean_latency_gpu_median_ms'),
            'core_forward_flops_observed_min':f.get('core_forward_flops_min'),
            'core_forward_flops_observed_max':f.get('core_forward_flops_max'),
            'full_all_kernel_flops_measured':f.get('full_all_kernel_flops_measured'),
            'pareto_final_phase_time_RMSE':e['pareto_RMSE_vs_measured_final_phase_time'],
            'pareto_parameter_count_RMSE':e['pareto_RMSE_vs_parameter_count'],
            'pareto_core_flops_RMSE':tables['flops_pareto'].get(model,{}).get('pareto_core_flops_RMSE'),
            'original_interval_score90':cal['mean_interval_score'],
            'MCS_pvalue':mcs['MCS_pvalue'],
            'MCS_retained_90':mcs['retained_at_alpha']})
        for kind in ['noise','outlier','missing','structural_break']:
            r['relative_RMSE_degradation_area_'+kind]=areas.get((model,kind),{}).get('normalized_degradation_area')
        results.append(r)
    if sum(int(r['R']) for r in results)!=81:
        raise ValueError('Expected 81 predictors')
    schema={
        'version':'e2-benchmark-current-comparison-v1',
        'sources':SOURCES,
        'rows':16,'dates':608,'predictors':81,
        'join_key':'model',
        'missing_value':'Empty CSV cell denotes not measured or not estimable; never zero.',
        'units_and_scopes':{
            'RMSE':'Original sigma7 units; mean of individual-seed RMSE, not ensemble RMSE.',
            'RMSE_seed_SD':'Sample SD of six initialization results conditional on one fixed chronology; blank for deterministic R=1.',
            'RMSE_seed_t_CI95_lower/upper':'Copied published marginal Student-t 95% interval over seed RMSE, not temporal or simultaneous confidence. No interval is recalculated.',
            'MAE':'Mean individual-seed original-unit MAE; deterministic models have one trajectory.',
            'sMAPE':'Fraction, not percentage. Multiply by100 only for an explicitly percentage-labelled display.',
            'coverage90':'Fraction averaged over original frozen per-seed intervals; nominal90%. Not posterior adaptive coverage.',
            'mean_interval_width90':'Original sigma7 units, same frozen intervals as A2.',
            'conditional_neural_mean/min/max_rank':'Published rank summaries of13 neural systems in six seed-ID vectors conditional on one chronology. No repeated deterministic rows and no new market replicates.',
            'display_position_by_RMSE':'Ascending display order across16 systems; descriptive only.',
            'actual_parameter_count':'Frozen predictive network only; SAE auxiliary decoders excluded. Classical comparable counts unmeasured.',
            'final_phase_seconds_mean_per_seed/SD/sum':'A100 phase wall time includes updates, eval risk and checkpoints; SAE includesAE1,AE2,supervised. Excludes entire HPO, outer controller and final scoring.',
            'final_training_peak_allocated_bytes_observed':'Process highwater since callers last reset, not isolated model peak. Scope copied per row.',
            'full_selection_seconds':'Unknown for all models; selected-recipe costs do not reconstruct full search.',
            'mean_latency_gpu_median_ms':'Mean of six per-seed medians; A100 CUDA-event forward, resident batch1 input,32warmup+100repetitions. Excludes transfer, preprocessing, weight load and service overhead. Not pooled median or95%CI.',
            'core_forward_flops_observed_min/max':'Observed min/max over three synthetic probes, not universal bounds. Dense matrix+conv algorithmic count,2FLOPs/MAC,B1eval; excludes FFT, activation, norm, pooling, bias and memory. TimesNet probes with initialization weights do not cover all data-dependent routes.',
            'pareto_*':'Previously published point-estimate fronts among13 neural systems only. No inferential dominance; latency frontier is a separate documentary plot from the same published points.',
            'relative_RMSE_degradation_area_*':'Copied normalized trapezoidal area of relative RMSE degradation over three severities. Fraction,lower better. Synthetic stressors on13 networks; no classical or external-market results.',
            'original_interval_score90':'Published mean interval score for original frozen90% intervals, not a classifier calibration metric.',
            'MCS_pvalue/retained_90':'Exploratory16-system T_R MCS90%,10000circular block14 draws on daily mean individual-seed MSE. Different estimand/family from primary practical RMSE tests; membership is not equivalence.'},
        'operations':{'new_model_fits':0,'new_inference_calls':0,'new_predictive_scores':0,'new_hypothesis_tests':0,'new_bootstrap_draws':0},
        'original_scientific_artifacts_modified':False}
    return results,schema


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    targets=[args.output_dir/'comparison_current.csv',args.output_dir/'comparison_schema.json']
    if any(p.exists() for p in targets):
        raise SystemExit('Refusing to overwrite an existing comparison. Use a new output directory.')
    result,schema=assemble(args.repository_root)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with targets[0].open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(result)
    targets[1].write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(result),'predictors':81,'new_model_fits':0,'new_hypothesis_tests':0}))


if __name__=='__main__':
    main()
