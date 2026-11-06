#!/usr/bin/env python3
"""Analyse closed development-only P0/OOF receipts, never fit or open raw data.

OOF Jan--Oct 2024 is the prespecified principal diagnostic gap cohort. Nov--Dec
P0 describes feasibility and conditional training variability only. All CIs are
nominal temporal intervals, not simultaneous intervals or final benchmark power.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import chi2, nct, t

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("TimeMixer", "iTransformer", "PatchTST", "GRU")
SEEDS = tuple(range(660101, 660106))
DELTA = 0.00025772842664981106
THRESHOLD = 0.024247163097617024
BLOCKS = (7, 14, 28)
B = 4999
MC_SEED = 20261106


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def json_read(path): return json.loads(Path(path).read_text())


def json_write(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(clean(v)) if isinstance(v, (dict, list, tuple)) else clean(v) for k, v in row.items()})


def calendar_for(phase):
    if phase == 'pilot': start, end = date(2024, 11, 1), date(2024, 12, 30)
    elif phase == 'oof': start, end = date(2024, 1, 1), date(2024, 10, 31)
    else: raise ValueError('Only pilot and oof development phases are permitted')
    return [(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)]


def boolean(value):
    if value is True or value == 'True': return True
    if value is False or value == 'False': return False
    raise ValueError('Explicit boolean True/False required')


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None: raise ValueError('Actual receipt times need an explicit timezone')
    return result


def validate_panel(rows, phase):
    """Every model/seed/day required exactly once; never compress the calendar."""
    days = calendar_for(phase); lookup = {v: i for i, v in enumerate(days)}
    prediction = np.full((4, 5, len(days)), np.nan)
    target = np.full_like(prediction, np.nan)
    persistence = np.full_like(prediction, np.nan)
    seen = set(); run_metadata = {}
    for row in rows:
        # Check date boundary before decoding any numeric outcome/prediction.
        day = row['outcome_day']
        if day not in lookup: raise ValueError('Outcome outside the fixed development cohort; 2025+ is forbidden')
        if row['model'] not in MODELS: raise ValueError('Unregistered model')
        seed = int(row['seed'])
        if seed not in SEEDS: raise ValueError('Unregistered seed')
        key = (row['model'], seed, day)
        if key in seen: raise ValueError('Duplicate model/seed/day')
        seen.add(key)
        origin = '2024-11-01' if phase == 'pilot' else day[:7] + '-01'
        cutoff = (date.fromisoformat(origin)-timedelta(days=8)).isoformat()
        if row['fit_origin'] != origin or row['fit_outcome_max'] != cutoff:
            raise ValueError('Unexpected fit origin or inherited eight-day information cutoff')
        for k in ['scaler_fit_outcome_max', 'configuration_selection_outcome_max']:
            if not row[k] or row[k] > cutoff: raise ValueError('Training/scaler/configuration used a later outcome')
        if row['run_id'] != f"{phase}_{origin}_{row['model']}_s{seed}": raise ValueError('Run ID does not match its phase/fold/model/seed')
        invariants = tuple(row[k] for k in ['checkpoint_sha256','checkpoint_closed_at_utc','scoring_started_at_utc','fit_outcome_max','scaler_fit_outcome_max','configuration_selection_outcome_max','mase_scale'])
        if row['run_id'] in run_metadata and run_metadata[row['run_id']] != invariants:
            raise ValueError('Checkpoint/scaler/closure metadata changes within one frozen run')
        run_metadata[row['run_id']] = invariants
        if row['scope'] != 'development_P0_not_final_benchmark' or row['source_stage'] != 'P0_diagnostic' or not boolean(row['eligible']):
            raise ValueError('Only eligible fixed-recipe P0 diagnostic predictions are allowed')
        if timestamp(row['checkpoint_closed_at_utc']) >= timestamp(row['scoring_started_at_utc']):
            raise ValueError('Checkpoint was not frozen before evaluation scoring')
        y, pred, lag = (float(row[k]) for k in ['y_true', 'y_pred', 'persistence_pred'])
        if not np.isfinite([y, pred, lag]).all() or min(y, pred, lag) < 0:
            raise ValueError('Volatility outcomes, predictions and causal persistence must be finite nonnegative')
        scale = float(row['mase_scale'])
        if not np.isfinite(scale) or scale <= 0: raise ValueError('Positive train-only MASE scale required')
        if boolean(row['high_volatility_known']) != (lag > THRESHOLD):
            raise ValueError('High regime does not match the fixed causal sigma7(t-1) threshold')
        i, j, k = MODELS.index(row['model']), SEEDS.index(seed), lookup[day]
        prediction[i, j, k], target[i, j, k], persistence[i, j, k] = pred, y, lag
    if len(seen) != prediction.size or not np.isfinite(prediction).all():
        raise ValueError('Incomplete model/seed/calendar panel; no dropping missing dates')
    if not np.array_equal(target, np.broadcast_to(target[0, 0], target.shape)):
        raise ValueError('Ground truth differs across paired models/seeds')
    if not np.array_equal(persistence, np.broadcast_to(persistence[0, 0], persistence.shape)):
        raise ValueError('Causal regime information differs across paired models/seeds')
    return {'phase': phase, 'days': days, 'predictions': prediction, 'targets': target[0, 0],
            'errors': prediction-target, 'high': persistence[0, 0] > THRESHOLD}


def mean_seed_rmse(squared_error):
    a = np.asarray(squared_error, dtype=float)
    if a.ndim != 3 or not a.size or not all(a.shape) or not np.isfinite(a).all() or np.any(a < 0): raise ValueError('Finite model×seed×date squared errors required')
    return np.sqrt(a.mean(axis=-1)).mean(axis=-1)


def region_gap(squared_error, high):
    a = np.asarray(squared_error, dtype=float); high = np.asarray(high, dtype=bool)
    if a.shape[-1] != len(high): raise ValueError('Regime mask must use the same dense calendar')
    if not high.any() or high.all(): return np.full(a.shape[0], np.nan)
    return mean_seed_rmse(a[..., high])-mean_seed_rmse(a[..., ~high])


def block_indices(n, length, rng):
    if not isinstance(length, (int, np.integer)) or not 1 <= length <= n: raise ValueError('Invalid calendar block length')
    starts = rng.integers(0, n, math.ceil(n/length))
    return ((starts[:, None]+np.arange(length)) % n).ravel()[:n]


def bootstrap_panel(squared_error, high, block=7, draws=B, seed=MC_SEED):
    a = np.asarray(squared_error, dtype=float); high = np.asarray(high, dtype=bool)
    mean_seed_rmse(a)
    if len(high) != a.shape[-1] or draws < 99: raise ValueError('Calendar mismatch or insufficient bootstrap draws')
    rng = np.random.default_rng(seed)
    rmse = np.empty((draws, a.shape[0])); gap = np.empty_like(rmse)
    for b in range(draws):
        idx = block_indices(len(high), block, rng)
        sampled = a[..., idx]
        rmse[b] = mean_seed_rmse(sampled)
        gap[b] = region_gap(sampled, high[idx])
    return {'RMSE': rmse, 'gap': gap, 'gain_vs_GRU': rmse[:, -1, None]-rmse[:, :-1]}


def interval(values):
    a = np.asarray(values, dtype=float); good = a[np.isfinite(a)]
    result = {'requested_draws': len(a), 'estimable_draws': len(good), 'failed_draws': len(a)-len(good),
              'minimum_estimable_fraction': .95, 'ci95': None, 'status': 'not_estimable'}
    if len(good) >= .95*len(a) and len(good):
        result.update(status='estimable', ci95=np.quantile(good, [.025, .975]).tolist())
    return result


def centered_upper_p(observed, bootstrap, boundary):
    b = np.asarray(bootstrap, dtype=float); b = b[np.isfinite(b)]
    if not np.isfinite(observed) or not len(b): return None
    return float((1+np.count_nonzero(b-observed >= observed-boundary))/(len(b)+1))


def holm_three(pvalues):
    if len(pvalues) != 3: raise ValueError('The registered family contains exactly three hypotheses')
    a = np.array([1. if p is None else p for p in pvalues], dtype=float)
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)): raise ValueError('Invalid p-value')
    order = np.argsort(a, kind='stable'); adjusted = np.empty(3)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(a[order]*np.arange(3, 0, -1)))
    return [None if p is None else float(v) for p, v in zip(pvalues, adjusted)]


def conditional_power(n, sd, epsilon=DELTA, alpha=.02/30):
    if n < 2 or sd <= 0 or epsilon <= 0 or not np.isfinite([sd, epsilon]).all(): return None
    return float(nct.sf(t.ppf(1-alpha, n-1), n-1, epsilon*math.sqrt(n)/sd))


def required_seeds(sd, epsilon=DELTA, alpha=.02/30, target=.8, minimum=5, cap=200):
    if not np.isfinite(sd) or sd <= 0:
        return {'status': 'not_estimable_zero_or_invalid_SD', 'required_R': None, 'power_at_cap': None}
    if not np.isfinite(epsilon) or epsilon <= 0 or not 0 < alpha < 1 or not 0 < target < 1 or minimum < 2 or cap < minimum:
        raise ValueError('Invalid power design')
    for n in range(minimum, cap+1):
        if conditional_power(n, sd, epsilon, alpha) >= target:
            return {'status': 'within_cap_conditional_only', 'required_R': n, 'power_at_required_R': conditional_power(n, sd, epsilon, alpha), 'power_at_cap': conditional_power(cap, sd, epsilon, alpha)}
    return {'status': 'required_R_exceeds_cap', 'required_R': None, 'required_R_lower_bound': cap+1, 'power_at_cap': conditional_power(cap, sd, epsilon, alpha)}


def seed_power_report(paired_gains):
    a = np.asarray(paired_gains, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all(): raise ValueError('Exactly five fresh paired-seed gains are required')
    sd = float(a.std(ddof=1)); df = 4
    sd_ci = [sd*math.sqrt(df/chi2.ppf(.975, df)), sd*math.sqrt(df/chi2.ppf(.025, df))] if sd > 0 else None
    return {'source': 'P0 November–December fixed-calendar paired training randomizations only',
        'n_pilot_seeds': 5, 'variance_df': df, 'observed_mean_gain': float(a.mean()), 'paired_difference_SD': sd,
        'SD_ci95_normal_seed_assumption': sd_ci,
        'null_boundary': DELTA, 'planned_alternative_gain': 2*DELTA, 'epsilon_beyond_null': DELTA,
        'one_sided_alpha': .02/30, 'multiplicity_denominator': 30, 'target_conditional_power': .8,
        'minimum_R': 5, 'cap_R': 200, 'point_SD_projection': required_seeds(sd),
        'optimistic_lower_SD_projection': required_seeds(sd_ci[0]) if sd_ci else None,
        'conservative_upper_SD_projection': required_seeds(sd_ci[1]) if sd_ci else None,
        'power_for_current_R5_under_planned_alternative': conditional_power(5, sd),
        'final_post_HPO_R_certified': False, 'joint_training_and_temporal_power_certified': False,
        'assumptions': 'Independent fitted seed randomizations; approximate normal paired RMSE differences conditional on this one calendar. SD uncertainty is large with df4. No observed gain is substituted for the planned alternative.',
        'limitations': 'P0 recipe differs from final post-HPO procedure. R is initialization replication, not independent markets or dates. Zero SD does not establish infinite precision or power.'}


def analyse_panel(panel, draws=B):
    phase = panel['phase']; errors = panel['errors']; sq = errors**2; high = panel['high']
    per_seed_rmse = np.sqrt(sq.mean(axis=-1)); point = mean_seed_rmse(sq)
    gaps = region_gap(sq, high)
    seeds = []
    for i, model in enumerate(MODELS):
        for j, seed in enumerate(SEEDS):
            h = float(np.sqrt(sq[i,j,high].mean())) if high.any() else None
            r = float(np.sqrt(sq[i,j,~high].mean())) if (~high).any() else None
            seeds.append({'phase': phase, 'model': model, 'seed': seed, 'n_origins': len(high), 'high_count': int(high.sum()),
                'rest_count': int((~high).sum()), 'RMSE': per_seed_rmse[i,j], 'MAE': np.abs(errors[i,j]).mean(),
                'MSE': sq[i,j].mean(), 'high_RMSE': h, 'rest_RMSE': r, 'high_minus_rest_RMSE': None if h is None or r is None else h-r})
    gap_rows, gains, aggregates = [], [], []
    for block in BLOCKS:
        boot = bootstrap_panel(sq, high, block, draws)
        local_rows = []
        for i, model in enumerate(MODELS):
            aggregates.append({'phase': phase, 'model': model, 'block_days': block, 'mean_seed_RMSE': point[i],
                'SD_seed_RMSE': per_seed_rmse[i].std(ddof=1), **interval(boot['RMSE'][:,i])})
            if i == 3: continue
            info = interval(boot['gap'][:,i])
            p = centered_upper_p(gaps[i], boot['gap'][:,i], DELTA) if info['status'] == 'estimable' and phase == 'oof' else None
            row = {'phase': phase, 'model': model, 'block_days': block, 'principal_diagnostic': phase == 'oof' and block == 7,
                'effect_RMSE_high_minus_rest': gaps[i], 'delta': DELTA, 'high_count': int(high.sum()), 'rest_count': int((~high).sum()),
                **info, 'p_one_sided_beyond_delta': p, 'p_Holm_family3': None, 'family_alpha': .01 if phase == 'oof' else None,
                'diagnostic_gap_supported': None, 'CI_is_simultaneous': False, 'phase_interpretation': 'principal OOF diagnostic' if phase == 'oof' else 'P0 descriptive only; no rejection decision'}
            local_rows.append(row)
            gains.append({'phase': phase, 'model': model, 'reference': 'fresh GRU/D7-c03', 'block_days': block,
                'gain_RMSE_GRU_minus_model': point[-1]-point[i], **interval(boot['gain_vs_GRU'][:,i]),
                'test_pvalue': None, 'interpretation': 'paired temporal descriptive gain; not a new final confirmatory comparison'})
        adjusted = holm_three([r['p_one_sided_beyond_delta'] for r in local_rows])
        for row, p in zip(local_rows, adjusted):
            row['p_Holm_family3'] = p
            if row['principal_diagnostic'] and p is not None:
                row['diagnostic_gap_supported'] = bool(p < .01 and row['effect_RMSE_high_minus_rest'] > DELTA)
        gap_rows.extend(local_rows)
    correlations = []
    for i, m in enumerate(MODELS):
        for k in range(i+1, 4):
            for j, seed in enumerate(SEEDS):
                a, b = errors[i,j], errors[k,j]
                corr = None if np.ptp(a) == 0 or np.ptp(b) == 0 else float(np.corrcoef(a,b)[0,1])
                correlations.append({'phase': phase, 'model_a': m, 'model_b': MODELS[k], 'seed': seed,
                    'n_dates': len(high), 'signed_error_Pearson_correlation': corr,
                    'interpretation': 'shared targets/history create dependence; correlation is not causal evidence or independent-market replication'})
    powers = {m: seed_power_report(per_seed_rmse[3]-per_seed_rmse[i]) for i, m in enumerate(MODELS[:3])} if phase == 'pilot' else None
    return {'seed_metrics': seeds, 'gap': gap_rows, 'comparative_gains': gains, 'aggregate_metrics': aggregates,
            'error_correlations': correlations, 'power': powers}


def verify_artifacts(directory, record):
    result = []
    for name, expected in record['artifact_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
            raise ValueError('Unsafe receipt artifact name')
        p = directory/name
        if p.is_symlink() or not p.is_file() or digest(p) != expected: raise ValueError('Closed artifact hash differs: '+name)
        result.append({'basename': name, 'sha256': expected})
    return result


def load_closed_phase(output, phase, plan, plan_hash, registration):
    base = output/(phase+'_scoring'); receipt_path = base/'completion.json'; receipt = json_read(receipt_path)
    jobs = 200 if phase == 'oof' else 20
    if (receipt.get('phase') != phase or receipt.get('plan_sha256') != plan_hash or receipt.get('scheduled_trajectories') != jobs
            or receipt.get('all_fit_closures_before_scoring') is not True or receipt.get('Bitcoin_test_results') != 0):
        raise ValueError('Scoring receipt is not the complete registered development phase')
    manifest = [{'kind': phase+'_scoring_receipt', 'basename': receipt_path.name, 'sha256': digest(receipt_path)}]
    manifest += [{'kind': phase+'_scoring_artifact', **item} for item in verify_artifacts(base, receipt)]
    if 'predictions.csv' not in receipt['artifact_sha256']: raise ValueError('No closed prediction manifest')
    with (base/'predictions.csv').open(newline='') as f: rows = list(csv.DictReader(f))
    if receipt.get('rows') != len(rows) or receipt.get('max_decoded_outcome') != calendar_for(phase)[-1]:
        raise ValueError('Scoring count or development boundary differs')
    panel = validate_panel(rows, phase)
    origins = [f'2024-{month:02d}-01' for month in range(1,11)] if phase == 'oof' else ['2024-11-01']
    by_run = {}
    for row in rows: by_run.setdefault(row['run_id'], row)
    score_times = {r['scoring_started_at_utc'] for r in by_run.values()}
    if len(score_times) != 1 or max(timestamp(r['checkpoint_closed_at_utc']) for r in by_run.values()) >= timestamp(next(iter(score_times))):
        raise ValueError('Every scheduled checkpoint must close before the first phase scoring')
    for origin in origins:
        for model in MODELS:
            for seed in SEEDS:
                folder = output/phase/origin/model/f'seed_{seed}'
                fit_receipt = json_read(folder/'completion.json')
                if any(fit_receipt.get(k) != v for k,v in {'plan_sha256':plan_hash,'phase':phase,'model':model,'seed':seed,'fit_origin':origin,'optimizer_fits':2,'evaluation_predictions_computed':False}.items()):
                    raise ValueError('Fit receipt identity or prediction-free closure differs')
                artifacts = verify_artifacts(folder, fit_receipt)
                required = {'checkpoint.pt','setup.json','preprocessing.json','split.json','history_inner.csv','history_refit.csv','started.json'}
                if set(fit_receipt['artifact_sha256']) != required: raise ValueError('Incomplete or unexpected closed fit artifact set')
                row = by_run[f'{phase}_{origin}_{model}_s{seed}']
                if (row['checkpoint_sha256'] != fit_receipt['artifact_sha256']['checkpoint.pt']
                        or row['checkpoint_closed_at_utc'] != fit_receipt['completed_at_utc']
                        or timestamp(registration['registered_at_utc']) >= timestamp(fit_receipt['completed_at_utc'])):
                    raise ValueError('Prediction/checkpoint/registration identity differs')
                setup = json_read(folder/'setup.json')
                for key in ['model','fit_origin','fit_outcome_max','scaler_fit_outcome_max','configuration_selection_outcome_max']:
                    if str(setup.get(key)) != row[key]: raise ValueError('Prediction metadata differs from frozen training setup')
                if setup.get('seed') != seed or setup.get('plan_sha256') != plan_hash or float(setup.get('mase_scale')) != float(row['mase_scale']):
                    raise ValueError('Frozen seed/plan/train-only scale does not match predictions')
                manifest.append({'kind': 'fit_receipt', 'run_id':row['run_id'], 'basename':'completion.json', 'sha256':digest(folder/'completion.json')})
                manifest += [{'kind':'fit_artifact', 'run_id':row['run_id'], **a} for a in artifacts]
    return panel, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path, help='Private registered runner output, outside repository')
    parser.add_argument('--output-dir', required=True, type=Path, help='Aggregate-only output; must be explicit')
    parser.add_argument('--design', type=Path, default=ROOT/'experiments/e2_remediation_20261106/pilot_design.json')
    parser.add_argument('--phases', nargs='+', choices=['pilot','oof'], default=['pilot','oof'])
    args = parser.parse_args(argv)
    if args.input_root.resolve().is_relative_to(ROOT): raise ValueError('Individual predictions/checkpoints must be private')
    allowed = ROOT/'book_e2/_static/remediation_pilot_results'
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT) and out != allowed: raise ValueError('Only the designated new aggregate results directory is permitted inside the repository')
    if out.exists(): raise ValueError('Use a fresh aggregate output directory; no silent result overwrite')
    plan = json_read(args.design); ph = digest(args.design)
    if tuple(plan['models']) != MODELS or tuple(plan['seeds']) != SEEDS or plan['margin_absolute'] != DELTA or plan['known_high_volatility_threshold'] != THRESHOLD:
        raise ValueError('Fixed model/seed/margin/regime contract differs')
    gap = plan['gap_analysis']
    if tuple(gap['block_lengths']) != BLOCKS or gap['bootstrap_draws'] != B or gap['Monte_Carlo_seed'] != MC_SEED:
        raise ValueError('Registered bootstrap settings differ')
    reg = json_read(args.input_root/'registration.json')
    if reg['immutable_inputs']['plan_sha256'] != ph or reg.get('dataset_fits_in_this_output_before_registration') != 0 or reg.get('Bitcoin_test_period_access') is not False:
        raise ValueError('Missing valid pre-fit development registration')
    if 'oof' in args.phases and plan['OOF']['enabled'] is not True: raise ValueError('OOF was not enabled before registration')
    outputs = []; source_manifest = [{'kind':'registration', 'basename':'registration.json', 'sha256':digest(args.input_root/'registration.json')}, {'kind':'design','basename':args.design.name,'sha256':ph}]
    for phase in dict.fromkeys(args.phases):
        panel, sources = load_closed_phase(args.input_root, phase, plan, ph, reg)
        outputs.append((phase, analyse_panel(panel))); source_manifest.extend(sources)
    out.mkdir(parents=True)
    for key in ['seed_metrics','aggregate_metrics','gap','comparative_gains','error_correlations']:
        write_csv(out/(key+'.csv'), [r for _, result in outputs for r in result[key]])
    json_write(out/'power.json', {'scope':'conditional P0 training replication planning, not final post-HPO or joint temporal power', 'models':next((r['power'] for phase,r in outputs if phase=='pilot'), None), 'OOF_power_not_substituted_for_P0':True})
    summary = {'scope':plan['scope'], 'phases':[phase for phase,_ in outputs], 'primary_gap_phase':'oof',
        'P0_role':'Nov–Dec feasibility/seed-variance only; descriptive gaps, no rejection decision',
        'models':MODELS,'seeds':SEEDS,'primary_metric':'mean across fitted-seed RMSE; no ensemble RMSE',
        'delta':DELTA,'high_volatility_threshold_fixed_train2023':THRESHOLD,
        'regime':'sigma7(t-1)>fixed q75, never evaluation target or added input channel',
        'bootstrap':{'draws':B,'seed':MC_SEED,'primary_block_days':7,'sensitivity_block_days':[14,28],'CI':'nominal percentile95, not simultaneous','all_models_and_seeds_resampled_jointly_by_date':True},
        'gap_family':{'hypotheses':3,'alpha':.01,'adjustment':'Holm','principal_decision':'OOF/block7 only; 14/28 are sensitivity'},
        'power':{'alpha':.02/30,'alternative_gain':2*DELTA,'epsilon':DELTA,'target_conditional':.8,'R_min':5,'R_cap':200,'final_post_HPO_power_established':False,'temporal_or_joint_power_established':False},
        'temporal_limit':'One Bitcoin chronology; R seeds are not independent markets. 60 or305 dates and T/block are not independent sample counts. Circular calendar blocks approximate serial dependence; overlapping target windows, regime imbalance and overlapping OOF training prefixes remain limitations.',
        'counts':{'P0_trajectories':20,'P0_optimizer_fits':40,'OOF_trajectories_if_enabled':200,'OOF_optimizer_fits_if_enabled':400},
        'new_fits_performed_by_analyzer':0,'test_2025_or_later_access':False,'final_Top3_or_HPO_selection_performed':False,
        'gap_guaranteed':False,'sources_considered_unseen_test':False,
        'interpretation':'Diagnostic development estimates only; nonsignificance is not equivalence. No result guarantee, no final benchmark or new confirmatory test.'}
    json_write(out/'summary.json', summary)
    json_write(out/'provenance.json', {'analyzer_basename':Path(__file__).name,'analyzer_sha256':digest(__file__),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'registered_source_commit':reg.get('source_commit'),
        'registered_development_sources':[{'basename':Path(k).name,'sha256':v} for k,v in reg['immutable_inputs'].get('development_sources',{}).items()],
        'inherited_scientific_sha256':reg['immutable_inputs'].get('inherited_scientific_sha256'),
        'input_receipts_and_artifacts':source_manifest,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'individual_predictions_or_weights_publicly_written':False,'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
