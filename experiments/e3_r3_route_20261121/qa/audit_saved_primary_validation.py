"""Independent recount of saved validation only; no RNG/MC/model execution."""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import json
import math

import numpy as np
from scipy import stats


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm_independent(p):
    values = list(map(float, p))
    ordered = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted = [None] * len(values)
    running = 0.
    for rank, index in enumerate(ordered):
        running = max(running, (len(values)-rank)*values[index])
        adjusted[index] = min(1., running)
    return np.array(adjusted)


def CP_independent(k, n):
    # Invert binomial tails by bisection rather than using the writer's beta PPF.
    if not 0 <= k <= n or n < 1:
        raise ValueError('Invalid binomial count')
    def inverse(function, target, increasing):
        lo, hi = 0., 1.
        for _ in range(64):
            middle = (lo+hi)/2
            if (function(middle) < target) == increasing:
                lo = middle
            else:
                hi = middle
        return (lo+hi)/2
    lower = 0. if k == 0 else inverse(lambda p: stats.binom.sf(k-1, n, p), .025, True)
    upper = 1. if k == n else inverse(lambda p: stats.binom.cdf(k, n, p), .025, False)
    return [lower, upper]


def audit(root, results, repo_root):
    runtime = root/'statistical_revision'
    plan = json.loads((runtime/'VALIDATION_PLAN.json').read_text())
    source_manifest = json.loads((runtime/'CODE_MANIFEST.json').read_text())
    for row in source_manifest['files']:
        assert digest(runtime/row['path']) == row['sha256'], row['path']
    receipt = json.loads((results/'execution_receipt.json').read_text())
    summary = json.loads((results/'validation_summary.json').read_text())
    decision = json.loads((results/'RESULTS_DECISION.json').read_text())
    assert receipt['source_manifest_SHA256'] == digest(runtime/'CODE_MANIFEST.json')
    assert receipt['plan_SHA256'] == digest(runtime/'VALIDATION_PLAN.json')
    assert receipt['raw_log_SHA256'] == digest(results/'per_replicate.jsonl.gz')
    with gzip.open(results/'per_replicate.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    completed = [x for x in records if 'result' in x]
    assert len(completed) == receipt['completed_replicates']
    assert len(records) == receipt['attempted_replicates']
    assert len({(x['case'], x['replicate']) for x in records}) == len(records)
    assert all(x['R3_accredited'] is False for x in [summary, receipt, decision])
    assert receipt['source_hashes_still_match'] and not receipt['source_implementation_changed_after_results']
    anchor = json.loads((runtime/'PUBLIC_ANCHOR.json').read_text())
    assert anchor['commit'] == receipt['public_anchor_commit']
    assert anchor['plan_SHA256'] == receipt['plan_SHA256']
    assert anchor['source_manifest_SHA256'] == receipt['source_manifest_SHA256']
    assert datetime.datetime.fromisoformat(anchor['observed_at_utc']) < \
           datetime.datetime.fromisoformat(receipt['started_at_utc'])
    fresh_q_hashes = {x['q_SHA256'] for x in completed}
    assert len(fresh_q_hashes) == len(completed)
    old_inputs = []
    for relative in (
        'experiments/e3_next_evidence_20261121/statistics/controlled_validation_run/per_replicate.jsonl.gz',
        'experiments/e3_statistics_remediation_20261121/calibration/validation_fresh/per_replicate.jsonl.gz',
    ):
        path = Path(repo_root) / relative
        with gzip.open(path, 'rt') as stream:
            older = [json.loads(line) for line in stream]
        old_q = {x['q_SHA256'] for x in older}
        assert fresh_q_hashes.isdisjoint(old_q)
        old_inputs.append({'path': str(path), 'SHA256': digest(path),
                           'saved_old_replicates': len(older),
                           'fresh_q_hashes_disjoint': True})
    cases = {x['id']: x for x in plan['cases']}
    summaries = {x['id']: x for x in summary['cases']}
    counts, max_holm_difference, max_P2_difference, max_CP_difference = [], 0., 0., 0.

    def check_rate(reported, events):
        nonlocal max_CP_difference
        n, k = len(events), int(sum(events))
        assert reported['replicates'] == n and reported['events'] == k
        assert reported['rate'] == k/n
        endpoints = CP_independent(k, n)
        difference = max(abs(a-b) for a, b in zip(endpoints, reported['CI95']))
        max_CP_difference = max(max_CP_difference, difference)
        assert difference < 2e-12

    for name, case in cases.items():
        saved = [x for x in completed if x['case'] == name]
        if not saved:
            continue
        report = summaries[name]
        assert len(saved) == report['completed_replicates']
        assert [x['replicate'] for x in saved] == list(range(len(saved)))
        truths = np.asarray(case['bases'][1:])-case['bases'][0]
        nulls = np.abs(truths) <= 1e-12
        all_best = {i for i, x in enumerate(case['bases']) if abs(x-min(case['bases'])) <= 1e-12}
        true_names = {'candidate' if i == 0 else f'rival_{i:02d}' for i in all_best}
        p1_event, p2_event, union_event, intersection_event, mcs_exclusion = [], [], [], [], []
        cover = [[[] for _ in range(2)] for _ in truths]
        for item in saved:
            result = item['result']
            assert item['analytic_longrun_theta'] == truths.tolist()
            raw, adjusted = [], []
            for j, row in enumerate(result['pairwise']):
                assert row['rival'] == f'rival_{j+1:02d}'
                raw.extend([row['P1_equal_two_sided_p'], row['P2_equal_two_sided_p']])
                adjusted.extend([row['P1_Holm'], row['P2_Holm']])
                assert abs(row['P1_equal_two_sided_p']*(plan['settings']['B']+1)-
                           round(row['P1_equal_two_sided_p']*(plan['settings']['B']+1))) < 1e-10
                if row['P2_supported']:
                    reference = 2*stats.t.sf(abs(row['theta_hat']/row['P2_SE']), case['R']-1)
                    max_P2_difference = max(max_P2_difference, abs(reference-row['P2_equal_two_sided_p']))
                    assert abs(reference-row['P2_equal_two_sided_p']) < 1e-13
                for k, method in enumerate(['P1', 'P2']):
                    ci = row[method+'_CI95']
                    cover[j][k].append(bool(ci is not None and ci[0] <= truths[j] <= ci[1]))
                    if method == 'P1' and ci is not None:
                        assert abs((ci[0]+ci[1])/2-row['theta_hat']) < 1e-12
            exact = holm_independent(raw)
            max_holm_difference = max(max_holm_difference, float(np.max(np.abs(exact-adjusted))))
            assert np.max(np.abs(exact-adjusted)) < 1e-14
            adj = exact.reshape(-1, 2)
            p1_event.append(bool(np.any(adj[nulls, 0] < .05)))
            p2_event.append(bool(np.any(adj[nulls, 1] < .05)))
            union_event.append(bool(np.any(adj[nulls] < .05)))
            intersection_event.append(bool(np.any(np.all(adj[nulls] < .05, axis=1))))
            mcs = result['MCS_bootstrap_t_frozen_control']
            mcs_exclusion.append(not true_names <= set(mcs['retained']))
        if nulls.any():
            check_rate(report['known_true_null_component_FWER_Holm38'], union_event)
            check_rate(report['known_true_null_FWER_by_component_within_same_Holm38']['P1'], p1_event)
            check_rate(report['known_true_null_FWER_by_component_within_same_Holm38']['P2'], p2_event)
            check_rate(report['known_true_null_two_method_intersection_event'], intersection_event)
        check_rate(report['MCS_bootstrap_t_frozen_control']['any_true_best_excluded'], mcs_exclusion)
        for j, row in enumerate(report['per_rival']):
            observed_bias = math.fsum(item['result']['pairwise'][j]['theta_hat']
                                      for item in saved)/len(saved)-truths[j]
            assert abs(observed_bias-row['bias_against_fixed_longrun_truth']) < 1e-12
            for k, method in enumerate(['P1', 'P2']):
                check_rate(row['methods'][method]['pointwise_CI95_coverage'], cover[j][k])
        counts.append({'case': name, 'replicates': len(saved), 'true_null_rivals': int(nulls.sum()),
                       'P1_FWER_events': sum(p1_event) if nulls.any() else None,
                       'P2_FWER_events': sum(p2_event) if nulls.any() else None,
                       'union_FWER_events': sum(union_event) if nulls.any() else None,
                       'intersection_events': sum(intersection_event) if nulls.any() else None,
                       'MCS_any_true_best_excluded': sum(mcs_exclusion),
                       'reported_FWER_screen': report.get('known_true_null_component_FWER_Holm38', {}).get('screen'),
                       'bias_range_against_population_truth': [
                           min(row['bias_against_fixed_longrun_truth'] for row in report['per_rival']),
                           max(row['bias_against_fixed_longrun_truth'] for row in report['per_rival'])],
                       'pointwise_CI95_coverage_range': {
                           method: [min(sum(cover[j][k])/len(saved) for j in range(len(truths))),
                                    max(sum(cover[j][k])/len(saved) for j in range(len(truths)))]
                           for k, method in enumerate(['P1', 'P2'])}})
    return {'kind': 'independent_read_only_saved_primary_validation_audit',
            'status': 'PASS_recount_and_math_checks_not_calibration_accreditation',
            'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_manifest_SHA256': digest(runtime/'CODE_MANIFEST.json'),
            'plan_SHA256': digest(runtime/'VALIDATION_PLAN.json'),
            'raw_log_SHA256': digest(results/'per_replicate.jsonl.gz'),
            'execution_receipt_SHA256': digest(results/'execution_receipt.json'),
            'public_anchor_commit': anchor['commit'],
            'public_anchor_observed_before_execution': True,
            'fresh_q_hashes_unique': True, 'preserved_old_input_checks': old_inputs,
            'completed_replicates_recounted': len(completed), 'cases': counts,
            'max_Holm_difference': max_holm_difference,
            'max_P2_reference_difference': max_P2_difference,
            'max_CP_difference_independent_binomial_inversion': max_CP_difference,
            'writer_scientific_decision': decision,
            'reviewer_new_MC_replicates': 0, 'reviewer_new_random_draws': 0,
            'reviewer_new_bootstrap_draws': 0, 'reviewer_new_model_fits': 0,
            'reviewer_new_market_predictions': 0, 'R3_accredited': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True, type=Path,
                        help='Preserved complete repository checkout for both legacy logs')
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an independent audit')
    root = Path(__file__).resolve().parents[1]
    result = audit(root, args.results, args.repo_root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'path': str(args.output), 'SHA256': digest(args.output),
                      'status': result['status']}))
