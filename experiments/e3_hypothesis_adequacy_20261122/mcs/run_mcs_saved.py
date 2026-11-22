"""Secondary fixed-cohort MCS from saved development forecasts only.

AI assistance declared. The referenced Tmax implementation is copied unchanged
from the existing project, with its own provenance in analysis_plan.json.
No market queries, neural imports, prediction calls or forecast fitting.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PINNED_PARSER = 'cc4af6b4428ad99616607b730683466568d9fb7b85d2875cdf6b7f32e08fbf3e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan['replicates'] != 1999 or plan['block_lengths'] != [7,14] or plan['alpha'] != .1:
        raise ValueError('Analysis settings differ from the fixed exploratory plan')
    if plan['candidate'] != 'input_only_mse' or plan['calendar_dates'] != 55:
        raise ValueError('Wrong candidate or calendar plan')
    return plan


def run(input_path, plan_path, output):
    if output.exists():
        raise ValueError('Results are immutable: supply a new output directory')
    plan = load_plan(plan_path)
    reference_path = HERE/'mcs_tmax_reference.py'
    parser_path = HERE.parent/'retrospective/audit_saved_adequacy.py'
    if sha(reference_path) != plan['source_reference']['sha256']:
        raise ValueError('Existing MCS reference bytes changed')
    if sha(parser_path) != PINNED_PARSER:
        raise ValueError('Saved-prediction integrity parser bytes changed')
    if sha(input_path) != plan['source_zip_sha256']:
        raise ValueError('Saved-prediction archive changed')
    parser = load_module('mcs_pinned_saved_integrity',parser_path)
    reference = load_module('mcs_pinned_reference',reference_path)
    sources=[Path(input_path),Path(plan_path),Path(__file__),reference_path,parser_path]
    inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sources]
    started = utc(); cpu = time.process_time(); clock = time.perf_counter()
    archive,original_plan,admission,dates,target,predictions,errors,trajectories = parser.load_saved(input_path)
    if set(predictions) != set(plan['source_recipes']) or len(predictions) != 8:
        raise ValueError('The complete eight-recipe pilot must be retained')
    if original_plan['seeds'] != plan['neural_seeds'] or len(trajectories) != 32:
        raise ValueError('Fixed fitted-seed cohort differs')
    if dates[0] != plan['calendar_start'] or dates[-1] != plan['calendar_end']:
        raise ValueError('Exposed calendar differs')
    panel = dict(original_model=plan['candidate'],targets=target,predictions=predictions,dates=dates)
    analyses=[]; membership=[]
    for block in plan['block_lengths']:
        settings = dict(mcs=dict(alpha=plan['alpha'],block_days=block,
            repetitions=plan['replicates'],seed=plan['rng_seed']))
        result = reference.run_mcs(panel,settings)
        result.update(scope=plan['scope'],primary_endpoint_warning=plan['primary_endpoint_warning'],
            implementation_sha256=sha(reference_path),analysis_plan_sha256=sha(plan_path))
        analyses.append(result)
        if result['status'] == 'completed' and result['stop_reason'].startswith('Equal predictive ability'):
            terminal_p=result['steps'][-1].get('p_stepwise') if result['steps'] else None
        else:
            terminal_p=None
        for recipe in result['initial_models']:
            included=recipe in result['included_models']
            membership.append(dict(recipe=recipe,block_length=block,
                status=result['status'],included_MCS90=included,
                elimination_stepwise_p=result['excluded_pvalues'].get(recipe),
                survivor_stepwise_p_lower_bound=terminal_p if included else None,
                individual_membership_p_computed=not included,
                mean_secondary_MSE=result['mean_loss'][recipe],
                candidate=recipe==plan['candidate'],
                interpretation='Survivor p not computed: lower bound only; inclusion is not equivalence' if included
                    else 'Actual sequential maximum-adjusted elimination p from the unchanged procedure'))
    elapsed=time.perf_counter()-clock; active=time.process_time()-cpu
    summary=dict(version=plan['version'],scope=plan['scope'],candidate=plan['candidate'],
        loss=plan['loss'],primary_endpoint_warning=plan['primary_endpoint_warning'],
        analysis_plan_sha256=sha(plan_path),source_zip_sha256=sha(input_path),
        date_start=dates[0],date_end=dates[-1],calendar_dates=len(dates),
        recipes=8,neural_recipe_count=len(original_plan['recipes']),
        neural_seed_ids=original_plan['seeds'],fixed_fitted_trajectories=len(trajectories),
        forecast_points=sum(p.size for p in predictions.values()),
        MCS_confidence=.9,MCS_alpha=.1,
        analyses=[dict(block_length=a['block_days'],status=a['status'],included=a['included_models'],
            excluded_pvalues=a['excluded_pvalues'],candidate_included=a['candidate_included'],
            steps=len(a['steps']),Tmax_initial=a['steps'][0]['Tmax'],
            initial_EPA_p=a['steps'][0]['p']) for a in analyses],
        calendar_bootstrap_index_draws=plan['replicates']*len(plan['block_lengths']),
        draws_each=plan['replicates'],minimum_plus_one_MC_p=1/(plan['replicates']+1),
        no_Holm_pairwise_decision=True,no_practical_RMSE_margin_test=True,
        not_third_primary_test=True,coverage_validation_performed=False,
        temporal_stability_or_stationarity_verified=False,
        seed_population_inference=False,candidate_selection_uncertainty_included=False,
        new_forecast_fits=0,new_neural_forwards=0,new_market_downloads=0,new_predictions=0,
        test_opened=False,R3_accredited=False,preregistration_claimed=False,
        AI_assistance_declared=True,limitations=plan['limits'])
    output.mkdir(parents=True)
    write=lambda name,value:(output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    write('summary.json',summary)
    write('mcs_full_results.json',analyses)
    with (output/'membership.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(membership[0]));writer.writeheader();writer.writerows(membership)
    receipt=dict(started_utc=started,finished_utc=utc(),wall_seconds_pipeline=elapsed,
        CPU_seconds_pipeline=active,timing_excludes='Python/module imports and environment preparation',
        source_inventory=inventory,all_sources_unchanged=all(sha(p)==r['sha256'] for p,r in zip(sources,inventory)),
        python=platform.python_version(),numpy=np.__version__,
        new_forecast_fits=0,new_model_inference=0,new_market_downloads=0,new_predictions=0,
        calendar_bootstrap_index_draws=3998,new_method_MC_validation=False,
        test_opened=False,preregistration_claimed=False,AI_assistance_declared=True)
    write('execution_receipt.json',receipt)
    write('OUTPUT_MANIFEST.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=HERE.parents[1]/'e3_input_only_candidate_20261121/inputs/public_results.zip')
    parser.add_argument('--plan',type=Path,default=HERE/'analysis_plan.json')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    args=parser.parse_args()
    run(args.input,args.plan,args.output)
