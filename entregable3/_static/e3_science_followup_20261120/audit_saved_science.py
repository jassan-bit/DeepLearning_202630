#!/usr/bin/env python3
"""Read-only audit of frozen V3 evidence. No model, fitting or resampling path."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b, message):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-15, message)


def run(root):
    started = time.monotonic()
    package = Path(__file__).resolve().parent
    inputs = load_json(package / 'input_manifest.json')
    for item in inputs['files']:
        source = root / item['path']
        content = source.read_bytes()
        require(len(content) == item['bytes'], 'size: ' + item['path'])
        require(hashlib.sha256(content).hexdigest() == item['sha256'], 'hash: ' + item['path'])

    pilot = root / 'experiments/e3_evidence_close_20261120/pilot/archive/payload'
    plan = load_json(pilot / 'plan.json')
    require(plan['total_required_fits'] == 30, 'pilot fits')
    require(plan['heuristic']['minimum_relative_RMSE_gain'] == .02, 'original pilot rule')
    require(len(plan['final_seeds']) == 3, 'pilot seed count')
    grouped = defaultdict(list)
    pilot_dates = set()
    for row in rows(pilot / 'assessment_predictions.csv'):
        pilot_dates.add(row['date'])
        if row['name'] in plan['neural_variants']:
            grouped[(row['name'], row['seed'])].append((float(row['target']), float(row['prediction'])))
    require(len(pilot_dates) == 55 and len(grouped) == 15, 'pilot prediction dimensions')
    rmse_pilot = defaultdict(list)
    for (name, seed), values in grouped.items():
        require(len(values) == 55, 'pilot incomplete trajectory')
        rmse_pilot[name].append(math.sqrt(math.fsum((y-p)**2 for y,p in values)/len(values)))
    mean_pilot = {name: math.fsum(values)/len(values) for name, values in rmse_pilot.items()}
    recipe_rows = rows(pilot / 'recipe_summary.csv')
    for row in recipe_rows:
        if row['name'] in mean_pilot:
            close(mean_pilot[row['name']], float(row['mean_RMSE']), 'pilot recipe mean')
    best_pilot_control = min((row for row in recipe_rows if row['name'] != 'atom_crps'), key=lambda row:float(row['mean_RMSE']))
    require(best_pilot_control['name'] == 'atom_mse_only', 'best pilot control retained')
    verdict = load_json(pilot / 'development_verdict.json')
    require(verdict['best_all_controls'] == best_pilot_control['name'], 'pilot verdict comparator')
    require(not verdict['continuation_heuristic_passed'], 'pilot verdict unchanged')
    adverse = mean_pilot['atom_crps']/mean_pilot['atom_mse_only'] - 1
    close(adverse, .02704311988801411, 'adverse pilot result')
    require(-adverse < .02, 'pilot continuation did not pass')

    transfer = root / 'experiments/e3_completion_20261120/reproduction/diagnostic_transfer'
    receipt = load_json(transfer / 'diagnostic_receipt.json')
    require(receipt['registered_decision_commit'] == '6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe', 'diagnostic decision identity')
    require(not receipt['confirmatory_claim'] and not receipt['R3_accredited'], 'diagnostic scope')
    require(not receipt['p_values_computed'], 'no diagnostic significance tests')
    require(receipt['new_model_fits'] == 0 and receipt['new_calibrations'] == 0, 'fixed-weight diagnostic')
    prediction_count = 0
    transfer_rmse = defaultdict(list)
    calendars = defaultdict(set)
    for name, sha256 in receipt['payload_sha256'].items():
        source = transfer / name
        require(hashlib.sha256(source.read_bytes()).hexdigest() == sha256, 'diagnostic receipt hash')
        values = rows(source)
        require(bool(values), 'empty diagnostic trajectory')
        require(len({v['date'] for v in values}) == len(values), 'duplicate diagnostic dates')
        dataset, recipe = values[0]['dataset'], values[0]['name']
        errors = []
        for row in values:
            y, p, floor = (float(row[key]) for key in ['target_sigma7','prediction_sigma7','floor_sigma7'])
            require(all(math.isfinite(x) for x in (y,p,floor)), 'nonfinite prediction')
            require(p + 1e-12 >= floor, 'floor violation')
            errors.append((y-p)**2)
            calendars[dataset].add(row['date'])
        prediction_count += len(values)
        transfer_rmse[(dataset,recipe)].append(math.sqrt(math.fsum(errors)/len(errors)))
    require(len(receipt['payload_sha256']) == 51 and prediction_count == 20400, 'diagnostic dimensions')
    mean_transfer = {key: math.fsum(value)/len(value) for key,value in transfer_rmse.items()}
    for row in rows(transfer / 'summary.csv'):
        close(mean_transfer[(row['dataset'],row['name'])], float(row['primary_mean_individual_seed_RMSE']), 'transfer summary arithmetic')
    best_controls = []
    for dataset in sorted(calendars):
        controls = {name: value for (panel,name),value in mean_transfer.items() if panel == dataset and name != 'atom_crps'}
        best = min(controls, key=controls.get)
        candidate = mean_transfer[(dataset,'atom_crps')]
        require(candidate > controls[best], 'diagnostic candidate does not beat best control')
        best_controls.append({'dataset':dataset, 'dates':len(calendars[dataset]), 'first_date':min(calendars[dataset]), 'last_date':max(calendars[dataset]), 'candidate_mean_individual_RMSE':candidate, 'best_observed_control':best, 'best_control_mean_individual_RMSE':controls[best], 'candidate_relative_error_increase':candidate/controls[best]-1, 'scope':'descriptive, exposed retrospective calendars; no inferential superiority claim'})

    power = root / 'experiments/e3_completion_20261120/protocol_power'
    planning = load_json(power / 'results/planning_receipt.json')
    draft = load_json(power / 'protocol.DRAFT.json')
    seed_rows = rows(power / 'results/seed_sensitivity.csv')
    calendar_rows = rows(power / 'results/calendar_sensitivity.csv')
    variances = rows(power / 'results/variance_inputs.csv')
    require((len(seed_rows),len(calendar_rows)) == (162,324), '486 conditional scenarios')
    require(planning['analytic_scenario_rows'] == len(seed_rows)+len(calendar_rows), 'scenario count')
    require(planning['Monte_Carlo_draws'] == 0, 'no power simulation')
    require(planning['pilot_seed_pairs'] == 3 and planning['guide_minimum_pilot_seed_pairs'] == 5, 'pilot variance sample insufficient')
    require(planning['main_rival_variances_available'] == 3 and planning['missing_main_rival_variances'] == 16, 'full-family variances missing')
    require(len(variances) == 6, 'six available variances including three ablations')
    require(planning['final_R'] is None and planning['final_calendar'] is None, 'final design unresolved')
    require(not planning['empirical_power_proven'], 'conditional planning only')
    require(not planning['additive_audit_SD_transferred'], 'no additive variance inheritance')
    family = draft['family']
    require(len(family['rivals']) == family['rival_count'] == 19, 'complete proposed family')
    require(family['main_component_count'] == 19*3 == 57, '57 test components')
    require(not draft['scientific_decision']['confirmation_go'], 'confirmation NO-GO')
    require(draft['freeze']['tag'] is None and draft['freeze']['commit'] is None, 'draft is not frozen')
    r1 = load_json(root / 'experiments/e3_r1_substantive_20261120/independent_review/technical_verdict.json')

    return {
        'schema_version':1,
        'status':'passed_read_only_scientific_followup',
        'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'base_main_commit':inputs['base_main_commit'],
        'input_files_hash_verified':len(inputs['files']),
        'pilot':{'preserved_fits':30,'assessment_trajectories':15,'assessment_dates':55,'candidate_relative_error_increase_vs_atom_mse_only':adverse,'original_threshold_unchanged':.02,'continuation_heuristic_passed':False},
        'transfer':{'saved_trajectories_checked':51,'saved_prediction_rows_checked':prediction_count,'all_summary_means_recomputed':21,'best_controls':best_controls,'independent_confirmation':False},
        'power':{'analytical_scenarios_verified':486,'Monte_Carlo_draws':0,'main_rivals':19,'test_components':57,'available_main_rival_variances':3,'missing_main_rival_variances':16,'pilot_seed_pairs':3,'guide_minimum_pilot_seed_pairs':5,'final_R':None,'final_calendar':None,'final_power_accredited':False},
        'decision':{'confirmation_GO':False,'R3_accredited':False,'automatic_ablation_promotion':False,'new_algorithm_proposed':False,'new_protocol_tag_created':False},
        'cost':{'wall_seconds':time.monotonic()-started,'python':platform.python_version(),'device':'CPU standard-library arithmetic and hashing','new_model_fits':0,'new_model_predictions':0,'new_calibrations':0,'statistical_resamples':0,'new_p_values':0,'GPU_seconds':0},
        'AI_assistance_declared':True,
        'independent_human_review_claimed':False,
        'R1_input_verdict_kind':r1.get('kind', r1.get('verdict', 'see immutable source verdict')),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=run(args.root.resolve())
    content=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.write_text(content,encoding='utf-8')
    print(content,end='')


if __name__ == '__main__':
    main()
