"""One frozen, bounded primary-repair campaign; no market operations."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy import stats
from revised_primary import Settings, run, holm, require
from dgp_revision import make_squared_errors, bases_for

RUNTIME_FILES = {'crossed_rmse_v2_frozen.py', 'revised_primary.py', 'dgp_revision.py',
                 'run_validation.py', 'test_deterministic.py', 'VALIDATION_PLAN.json',
                 'LEGACY_REFERENCE.json', 'deterministic_diagnosis.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False)+'\n')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def preflight(root, anchor_path):
    root = Path(root)
    plan = json.loads((root/'VALIDATION_PLAN.json').read_text())
    manifest = json.loads((root/'CODE_MANIFEST.json').read_text())
    anchor = json.loads(Path(anchor_path).read_text())
    require(anchor.get('status') == 'new_primary_code_and_plan_published_before_validation'
            and anchor.get('observed_by_root_before_validation') is True,
            'Actual immutable public anchor observed by root required')
    commit = anchor.get('commit', '')
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit)
            and commit != '0'*40, 'Exact nonzero public commit required')
    observed = datetime.datetime.fromisoformat(anchor['observed_at_utc'])
    require(observed.tzinfo is not None and observed <= datetime.datetime.now(datetime.timezone.utc),
            'Prior timezone-aware observation timestamp required')
    require(anchor.get('plan_SHA256') == sha(root/'VALIDATION_PLAN.json')
            and anchor.get('source_manifest_SHA256') == sha(root/'CODE_MANIFEST.json'),
            'Public source or plan hash differs')
    require(manifest.get('kind') == 'frozen_before_new_primary_validation'
            and {r['path'] for r in manifest['files']} == RUNTIME_FILES
            and len(manifest['files']) == len(RUNTIME_FILES), 'Exact eight-file runtime inventory required')
    for row in manifest['files']:
        path = Path(row['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe runtime path')
        payload = (root/path).read_bytes()
        require(len(payload) == row['bytes'] and hashlib.sha256(payload).hexdigest() == row['sha256'],
                'Frozen source changed: '+str(path))
    require(plan['planned_replicates'] == sum(c['replicates'] for c in plan['cases']) == 1664
            and len(plan['cases']) == 7 and [c['replicates'] for c in plan['cases']] == [256]*6+[128],
            'Fixed seven-case replicate allocation changed')
    require(plan['validation_seed'] == 2026112121
            and plan['settings'] == {'B': 999, 'mean_block': 12., 'hac_lag': 8,
                                    'alpha': .05, 'mcs_alpha': .10, 'batch': 16}
            and plan['max_wall_seconds'] == 600 and plan['threads'] == 2
            and plan['planned_crossed_bootstrap_draws'] == 1662336,
            'Frozen seeds/settings/budget changed')
    require(os.environ.get('OPENBLAS_NUM_THREADS') == '2'
            and os.environ.get('OMP_NUM_THREADS') == '2', 'Exactly two BLAS/OMP threads required')
    return plan, manifest, anchor


def CP(k, n):
    require(n > 0 and 0 <= k <= n, 'Invalid binomial count')
    return {'events': int(k), 'replicates': int(n), 'rate': float(k/n),
            'CI95': [0. if k == 0 else float(stats.beta.ppf(.025, k, n-k+1)),
                     1. if k == n else float(stats.beta.ppf(.975, k+1, n-k))],
            'simultaneous_interval': False}


def rate(values):
    values = np.asarray(values, dtype=bool)
    return CP(int(values.sum()), len(values))


def screen(summary, nominal, unsupported=False):
    lo, hi = summary['CI95']
    status = ('inflation_detected' if lo > nominal else
              ('exploratory_tolerance_screen_passed' if hi <= nominal+.03 else 'inconclusive'))
    if unsupported:
        status = 'inconclusive_unsupported_replicates'
    return {**summary, 'nominal': nominal, 'exploratory_tolerance': .03, 'screen': status,
            'exact_nominal_level_certified': False, 'market_calibration_certified': False}


def coverage_screen(summary):
    lo, hi = summary['CI95']
    status = ('undercoverage_detected' if hi < .95 else
              ('exploratory_tolerance_screen_passed' if lo >= .92 else 'inconclusive'))
    return {**summary, 'nominal': .95, 'exploratory_tolerance': .03, 'screen': status,
            'simultaneous_coverage_claimed': False}


def arrays(records, p1, p2):
    return np.array([[[p[p1], p[p2]] for p in x['result']['pairwise']] for x in records])


def summarize(case, records):
    bases = bases_for(case)
    truth = bases[1:]-bases[0]
    null = np.abs(truth) <= 1e-12
    names = ['candidate']+[f'rival_{i:02d}' for i in range(1, case['M'])]
    out = {'id': case['id'], 'planned_replicates': case['replicates'],
           'completed_replicates': len(records), 'complete': len(records) == case['replicates'],
           'shape': [case['R'], case['T'], case['M']], 'analytic_longrun_theta': truth.tolist(),
           'true_null_rivals': [names[i+1] for i in np.flatnonzero(null)],
           'global_EPA_null_true': bool(null.all()), 'practical_delta': case['delta'],
           'per_rival': [], 'R3_accredited': False, 'final_market_power_certified': False}
    if not records:
        return out
    raw = arrays(records, 'P1_equal_two_sided_p', 'P2_equal_two_sided_p')
    adj = arrays(records, 'P1_Holm', 'P2_Holm')
    control = arrays(records, 'P1_basic_frozen_control_Holm38', 'P2_frozen_control_Holm38')
    estimates = np.array([[p['theta_hat'] for p in x['result']['pairwise']] for x in records])
    cis = np.array([[[p[m+'_CI95'] or [np.nan, np.nan] for m in ['P1', 'P2']]
                     for p in x['result']['pairwise']] for x in records])
    supported = np.isfinite(cis).all(axis=-1)
    covered = supported & (cis[..., 0] <= truth[None, :, None]) & (truth[None, :, None] <= cis[..., 1])
    both = np.all(adj < .05, axis=2)
    practical = both & (estimates > case['delta']) & np.all(cis[..., 0] > case['delta'], axis=2)
    for j, name in enumerate(names[1:]):
        row = {'rival': name, 'true_theta': float(truth[j]),
               'bias_against_fixed_longrun_truth': float(estimates[:, j].mean()-truth[j]),
               'empirical_SD_theta': float(estimates[:, j].std(ddof=1)) if len(records) > 1 else None,
               'methods': {}}
        for k, method in enumerate(['P1', 'P2']):
            widths = cis[:, j, k, 1]-cis[:, j, k, 0]
            row['methods'][method] = {
                'raw_rejection': rate(raw[:, j, k] < .05),
                'Holm38_rejection': rate(adj[:, j, k] < .05),
                'pointwise_CI95_coverage': coverage_screen(rate(covered[:, j, k])),
                'unsupported_intervals': int((~supported[:, j, k]).sum()),
                'missing_interval_counts_as_noncoverage': True,
                'mean_CI_width': float(widths[supported[:, j, k]].mean()) if supported[:, j, k].any() else None,
                'positive_and_lowerCI_above_delta': rate((estimates[:, j] > case['delta']) &
                                                       (cis[:, j, k, 0] > case['delta']))}
        row['two_method_intersection_Holm38'] = rate(both[:, j])
        row['two_method_practical_subset_decision'] = rate(practical[:, j])
        out['per_rival'].append(row)
    out['simultaneous_all19_by2_pointwiseCI_coverage_diagnostic'] = rate(np.all(covered, axis=(1, 2)))
    out['simultaneous_CI95_claimed'] = False
    out['unsupported_replicates_by_method'] = {
        method: int(np.any(~supported[:, :, k], axis=1).sum()) for k, method in enumerate(['P1', 'P2'])}
    if null.any():
        new_event = np.any(adj[:, null, :] < .05, axis=(1, 2))
        old_event = np.any(control[:, null, :] < .05, axis=(1, 2))
        unsupported_new = bool(np.any(~supported))
        unsupported_old = any(not p['P2_supported'] for x in records for p in x['result']['pairwise'])
        out['known_true_null_component_FWER_Holm38'] = screen(rate(new_event), .05, unsupported_new)
        out['old_basic_control_known_true_null_FWER_Holm38'] = screen(rate(old_event), .05, unsupported_old)
        out['known_true_null_FWER_by_component_within_same_Holm38'] = {
            method: rate(np.any(adj[:, null, k] < .05, axis=1)) for k, method in enumerate(['P1', 'P2'])}
        out['known_true_null_two_method_intersection_event'] = rate(np.any(both[:, null], axis=1))
        out['known_true_null_directional_practical_subset_event'] = rate(np.any(practical[:, null], axis=1))
        old_only = int((old_event & ~new_event).sum())
        new_only = int((new_event & ~old_event).sum())
        discordant = old_only+new_only
        diag_p = (1. if discordant == 0 or unsupported_new or unsupported_old else
                  float(stats.binomtest(old_only, discordant, .5).pvalue))
        out['paired_old_basic_mechanism_diagnostic'] = {
            'old_only_FWER': old_only, 'new_only_FWER': new_only,
            'exact_McNemar_two_sided_p': diag_p, 'unsupported': unsupported_new or unsupported_old,
            'Holm6_p': None, 'causal_explanation_or_superiority_certified': False}
    if (~null).any():
        out['all_nonnull_two_method_joint_Holm38_rejection'] = rate(np.all(adj[:, ~null, :] < .05, axis=(1, 2)))
        out['all_nonnull_two_method_practical_subset_decision'] = rate(np.all(practical[:, ~null], axis=1))
    mcs = [x['result']['MCS_bootstrap_t_frozen_control'] for x in records]
    best = {names[i] for i in np.flatnonzero(np.isclose(bases, bases.min(), atol=1e-12, rtol=0))}
    rejected = [x.get('initial_global_p') is not None and x['initial_global_p'] < .10 for x in mcs]
    unsupported_mcs = sum(x.get('initial_global_p') is None for x in mcs)
    global_rate = rate(rejected)
    out['MCS_bootstrap_t_frozen_control'] = {
        'initial_global_rejection': screen(global_rate, .10, bool(unsupported_mcs)) if null.all() else global_rate,
        'true_best_models': sorted(best),
        'all_true_best_retained': rate([best <= set(x['retained']) for x in mcs]),
        'any_true_best_excluded': rate([not best <= set(x['retained']) for x in mcs]),
        'candidate_retained': rate(['candidate' in x['retained'] for x in mcs]),
        'unique_candidate_set': rate([x['retained'] == ['candidate'] for x in mcs]),
        'false_unique_candidate': rate([x['retained'] == ['candidate'] and best != {'candidate'} for x in mcs]),
        'unsupported_replicates': unsupported_mcs, 'third_pairwise_p_provided': False}
    out['complete_three_method_power_computed'] = False
    return out


def decision(summaries, complete):
    screens = [s.get('known_true_null_component_FWER_Holm38', {}).get('screen')
               for s in summaries if s.get('true_null_rivals')]
    if not complete:
        status = 'incomplete_no_family_approval'
    elif any(x == 'inflation_detected' for x in screens):
        status = 'new_development_family_not_approved_inflation_detected'
    elif len(screens) == 6 and all(x == 'exploratory_tolerance_screen_passed' for x in screens):
        status = 'bounded_component_FWER_tolerance_screens_passed_not_accreditation'
    else:
        status = 'new_development_family_inconclusive_not_approved'
    return {'status': status, 'complete': complete, 'known_true_null_FWER_screens': screens,
            'coverage_reported_separately_not_guaranteed_by_FWER_screens': True,
            'not_all_claimed_endpoints_accredited_by_FWER_tolerance_pass': True,
            'primary_market_admissibility_certified': False, 'exact_nominal_level_certified': False,
            'third_primary_method_accredited': False, 'complete_three_method_power_computed': False,
            'confirmed_valid_for_market': False, 'confirmation_allowed': False,
            'R3_accredited': False, 'new_market_fits': 0, 'new_market_predictions': 0,
            'new_market_forwards': 0, 'V3_NO_GO_unchanged': True, 'V4_abandonment_unchanged': True}


def final_state(completed, errors, source_match):
    complete = completed == 1664 and not errors and source_match
    status = 'completed' if complete else ('error' if errors or not source_match else 'resource_stop')
    return complete, status


def execute(root, output, anchor_path):
    root, output = Path(root), Path(output)
    plan, manifest, anchor = preflight(root, anchor_path)
    require(not output.exists(), 'New output directory required; no overwrite or favorable rerun')
    output.mkdir(parents=True)
    started, start = utc(), time.monotonic()
    settings = Settings(**plan['settings'])
    receipt = {'status': 'running', 'started_at_utc': started,
               'plan_SHA256': sha(root/'VALIDATION_PLAN.json'),
               'source_manifest_SHA256': sha(root/'CODE_MANIFEST.json'),
               'public_anchor_commit': anchor['commit'], 'planned_replicates': 1664,
               'planned_crossed_bootstrap_draws': 1662336, 'completed_replicates': 0,
               'attempted_replicates': 0, 'errors': [], 'new_market_fits': 0,
               'new_market_predictions': 0, 'new_market_forwards': 0, 'R3_accredited': False,
               'AI_assistance_declared': True, 'human_execution_claimed': False,
               'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                           'scipy': scipy.__version__, 'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
                           'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')}}
    write(output/'execution_receipt.json', receipt)
    summaries, stop = [], False
    with (output/'per_replicate.jsonl.gz').open('wb') as rawfile, gzip.GzipFile(
            filename='', mode='wb', fileobj=rawfile, mtime=0) as log:
        for index, case in enumerate(plan['cases']):
            records = []
            for replicate in range(case['replicates']):
                if time.monotonic()-start >= plan['max_wall_seconds']:
                    stop = True
                    break
                receipt['attempted_replicates'] += 1
                item = {'case': case['id'], 'replicate': replicate, 'stage': 'new_primary_controlled_validation'}
                try:
                    dgp_rng = np.random.default_rng(np.random.SeedSequence([plan['validation_seed'], index, replicate, 0]))
                    test_rng = np.random.default_rng(np.random.SeedSequence([plan['validation_seed'], index, replicate, 1]))
                    q, truth, bases = make_squared_errors(case, dgp_rng)
                    names = ['candidate']+[f'rival_{i:02d}' for i in range(1, case['M'])]
                    item.update({'q_SHA256': hashlib.sha256(q.tobytes(order='C')).hexdigest(),
                                 'analytic_longrun_theta': truth.tolist(), 'bases': bases.tolist(),
                                 'result': run(q, names, 0, test_rng, settings)})
                    records.append(item)
                    receipt['completed_replicates'] += 1
                except Exception as error:
                    item['error'] = {'type': type(error).__name__, 'message': str(error)}
                    receipt['errors'].append({'case': case['id'], 'replicate': replicate, **item['error']})
                    stop = True
                log.write((json.dumps(item, sort_keys=True, allow_nan=False)+'\n').encode())
                log.flush()
                if stop:
                    break
            summary = summarize(case, records)
            summaries.append(summary)
            write(output/(case['id']+'_summary.json'), summary)
            receipt['elapsed_seconds'] = time.monotonic()-start
            write(output/'execution_receipt.json', receipt)
            print(json.dumps({'case': case['id'], 'completed': len(records),
                              'planned': case['replicates'], 'total_completed': receipt['completed_replicates'],
                              'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)
            if stop:
                break
    comparisons = [s['paired_old_basic_mechanism_diagnostic'] for s in summaries
                   if 'paired_old_basic_mechanism_diagnostic' in s]
    if len(comparisons) == 6:
        adjusted = holm([x['exact_McNemar_two_sided_p'] for x in comparisons])
        for x, p in zip(comparisons, adjusted):
            x['Holm6_p'] = float(p)
    for summary in summaries:
        write(output/(summary['id']+'_summary.json'), summary)
    source_match = all(sha(root/r['path']) == r['sha256'] for r in manifest['files'])
    complete, status = final_state(receipt['completed_replicates'], receipt['errors'], source_match)
    failed_attempts = receipt['attempted_replicates']-receipt['completed_replicates']
    receipt.update({'status': status,
                    'finished_at_utc': utc(), 'elapsed_seconds': time.monotonic()-start,
                    'actual_crossed_bootstrap_draws': None if failed_attempts else receipt['completed_replicates']*settings.B,
                    'completed_replicate_bootstrap_draws': receipt['completed_replicates']*settings.B,
                    'failed_attempt_bootstrap_draws_exact': None if failed_attempts else 0,
                    'failed_attempt_bootstrap_draws_upper_bound': failed_attempts*settings.B,
                    'control_extra_bootstrap_draws': 0, 'source_implementation_changed_after_results': not source_match,
                    'source_hashes_still_match': source_match,
                    'source_integrity_error': None if source_match else 'Frozen source or plan differs at final integrity check; no approval',
                    'raw_log_SHA256': sha(output/'per_replicate.jsonl.gz')})
    write(output/'execution_receipt.json', receipt)
    overall = {'status': receipt['status'], 'complete': complete, 'cases': summaries,
               'public_anchor_commit': anchor['commit'],
               'source_manifest_SHA256': receipt['source_manifest_SHA256'],
               'three_primary_pairwise_tests_accredited': False, 'R3_accredited': False,
               'new_market_fits': 0, 'new_market_predictions': 0, 'new_market_forwards': 0}
    write(output/'validation_summary.json', overall)
    write(output/'RESULTS_DECISION.json', decision(summaries, complete))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--anchor', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    anchor = args.anchor or args.root/'PUBLIC_ANCHOR.json'
    if args.execute:
        require(args.output is not None, '--output required with --execute')
        print(json.dumps(execute(args.root, args.output, anchor), sort_keys=True), flush=True)
    else:
        plan, _, public = preflight(args.root, anchor)
        print(json.dumps({'status': 'preflight_passed_without_execution',
                          'planned_replicates': plan['planned_replicates'],
                          'public_anchor_commit': public['commit'], 'new_validation_replicates': 0}), flush=True)
