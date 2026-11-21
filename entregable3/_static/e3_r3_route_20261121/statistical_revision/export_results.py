"""Export only a closed, hash-checked bounded campaign; zero random draws."""
import argparse
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path

TERMINAL = {'completed', 'resource_stop', 'error'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def holm(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out, previous = [None]*len(values), 0.
    for rank, index in enumerate(order):
        previous = max(previous, (len(values)-rank)*values[index])
        out[index] = min(1., previous)
    return out


def load_closed(root):
    root = Path(root)
    output = root/'validation_new'
    receipt_path = output/'execution_receipt.json'
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    require(receipt.get('status') in TERMINAL and receipt.get('finished_at_utc'),
            'Campaign is not closed; no unfinished-log interpretation is allowed')
    plan, manifest = read_json(root/'VALIDATION_PLAN.json'), read_json(root/'CODE_MANIFEST.json')
    anchor = read_json(root/'PUBLIC_ANCHOR.json')
    require(receipt['plan_SHA256'] == sha(root/'VALIDATION_PLAN.json') == anchor['plan_SHA256'],
            'Plan hash mismatch')
    require(receipt['source_manifest_SHA256'] == sha(root/'CODE_MANIFEST.json') == anchor['source_manifest_SHA256'],
            'Source manifest hash mismatch')
    require(receipt['public_anchor_commit'] == anchor['commit'], 'Anchor commit mismatch')
    for row in manifest['files']:
        path = root/row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Frozen source changed: '+row['path'])
    require(receipt.get('source_hashes_still_match') is True
            and receipt.get('source_implementation_changed_after_results') is False,
            'Frozen campaign integrity was not maintained')
    raw_path = output/'per_replicate.jsonl.gz'
    require(sha(raw_path) == receipt['raw_log_SHA256'], 'Closed raw log hash mismatch')
    overall, decision = read_json(output/'validation_summary.json'), read_json(output/'RESULTS_DECISION.json')
    require(overall['status'] == receipt['status'], 'Summary/receipt terminal status differs')
    cases = {case['id']: case for case in plan['cases']}
    records, errors, seen_q = {name: [] for name in cases}, [], set()
    max_holm_error = 0.
    attempted, success = 0, 0
    with gzip.open(raw_path, 'rt') as handle:
        for line in handle:
            item = json.loads(line)
            attempted += 1
            name = item['case']
            require(name in cases, 'Unplanned case in raw log')
            case = cases[name]
            require(item['replicate'] == len(records[name]), 'Missing/duplicated/out-of-order successful replicate')
            if 'error' in item:
                errors.append(item)
                continue
            require(item['analytic_longrun_theta'] == [b-case['bases'][0] for b in case['bases'][1:]],
                    'Population truth changed in raw log')
            require(item['q_SHA256'] not in seen_q, 'Duplicate synthetic data hash')
            seen_q.add(item['q_SHA256'])
            result = item['result']
            require(result['shape'] == [case['R'], case['T'], case['M']]
                    and len(result['pairwise']) == case['M']-1, 'Raw shape or family slots differ')
            require(result['bootstrap_studentizers_recomputed'] == plan['settings']['B'],
                    'Bootstrap studentizer count differs')
            require(result['R3_accredited'] is False and result['third_primary_method_accredited'] is False
                    and result['new_market_fits'] == result['new_market_predictions'] == 0,
                    'Raw output incorrectly claims accreditation or market execution')
            rows = result['pairwise']
            new_raw = [p for row in rows for p in [row['P1_equal_two_sided_p'], row['P2_equal_two_sided_p']]]
            old_raw = [p for row in rows for p in [row['P1_basic_frozen_control_p'], row['P2_equal_two_sided_p']]]
            new_adjusted, old_adjusted = holm(new_raw), holm(old_raw)
            observed_new = [p for row in rows for p in [row['P1_Holm'], row['P2_Holm']]]
            observed_old = [p for row in rows for p in [row['P1_basic_frozen_control_Holm38'], row['P2_frozen_control_Holm38']]]
            max_holm_error = max(max_holm_error, *(abs(a-b) for a, b in zip(new_adjusted, observed_new)),
                                 *(abs(a-b) for a, b in zip(old_adjusted, observed_old)))
            records[name].append(item)
            success += 1
    require(max_holm_error <= 1e-14, 'Holm38 recount differs')
    require(success == receipt['completed_replicates'] and attempted == receipt['attempted_replicates'],
            'Closed raw replicate recount differs')
    require(len(errors) == len(receipt['errors']), 'Error recount differs')
    require(receipt['completed_replicate_bootstrap_draws'] == success*plan['settings']['B'],
            'Completed-replicate draw recount differs')
    if not errors:
        require(receipt['actual_crossed_bootstrap_draws'] == success*plan['settings']['B'],
                'Total bootstrap draw recount differs')
    require(receipt_bytes == receipt_path.read_bytes(), 'Receipt changed during closed-log export')
    return plan, receipt, overall, decision, records, max_holm_error


def scalar_rate(summary, prefix):
    if not summary:
        return {prefix+'_events': None, prefix+'_n': None, prefix+'_rate': None,
                prefix+'_CP95_lower': None, prefix+'_CP95_upper': None, prefix+'_screen': None}
    return {prefix+'_events': summary['events'], prefix+'_n': summary['replicates'],
            prefix+'_rate': summary['rate'], prefix+'_CP95_lower': summary['CI95'][0],
            prefix+'_CP95_upper': summary['CI95'][1], prefix+'_screen': summary.get('screen')}


def check_event(summary, count, n, label):
    require(summary['events'] == count and summary['replicates'] == n, 'Endpoint recount differs: '+label)


def recount_case(case, summary, records):
    if not records:
        return
    truth = [b-case['bases'][0] for b in case['bases'][1:]]
    null = [abs(theta) <= 1e-12 for theta in truth]
    fwer = old_fwer = intersection = practical_null = 0
    mcs_global = all_best = false_unique = nonnull_joint = nonnull_practical = 0
    best = {i for i, b in enumerate(case['bases']) if abs(b-min(case['bases'])) <= 1e-12}
    name_to_index = {'candidate': 0, **{f'rival_{i:02d}': i for i in range(1, case['M'])}}
    per_method_counts = [[0, 0] for _ in truth]
    per_method_coverage = [[0, 0] for _ in truth]
    for item in records:
        rows = item['result']['pairwise']
        both = [row['P1_Holm'] < .05 and row['P2_Holm'] < .05 for row in rows]
        practical = [both[j] and row['theta_hat'] > case['delta']
                     and row['P1_CI95'] is not None and row['P2_CI95'] is not None
                     and row['P1_CI95'][0] > case['delta'] and row['P2_CI95'][0] > case['delta']
                     for j, row in enumerate(rows)]
        fwer += any(null[j] and (row['P1_Holm'] < .05 or row['P2_Holm'] < .05) for j, row in enumerate(rows))
        old_fwer += any(null[j] and (row['P1_basic_frozen_control_Holm38'] < .05 or row['P2_frozen_control_Holm38'] < .05)
                        for j, row in enumerate(rows))
        intersection += any(null[j] and both[j] for j in range(len(null)))
        practical_null += any(null[j] and practical[j] for j in range(len(null)))
        if not all(null):
            nonnull_joint += all(both[j] for j in range(len(null)) if not null[j])
            nonnull_practical += all(practical[j] for j in range(len(null)) if not null[j])
        for j, row in enumerate(rows):
            for k, method in enumerate(['P1', 'P2']):
                per_method_counts[j][k] += row[method+'_Holm'] < .05
                ci = row[method+'_CI95']
                per_method_coverage[j][k] += ci is not None and ci[0] <= truth[j] <= ci[1]
        mcs = item['result']['MCS_bootstrap_t_frozen_control']
        retained = {name_to_index[name] for name in mcs['retained']}
        mcs_global += mcs.get('initial_global_p') is not None and mcs['initial_global_p'] < .10
        all_best += best <= retained
        false_unique += retained == {0} and best != {0}
    n = len(records)
    if any(null):
        for key, count in [('known_true_null_component_FWER_Holm38', fwer),
                           ('old_basic_control_known_true_null_FWER_Holm38', old_fwer),
                           ('known_true_null_two_method_intersection_event', intersection),
                           ('known_true_null_directional_practical_subset_event', practical_null)]:
            check_event(summary[key], count, n, case['id']+'/'+key)
    if not all(null):
        check_event(summary['all_nonnull_two_method_joint_Holm38_rejection'], nonnull_joint, n, 'joint2method')
        check_event(summary['all_nonnull_two_method_practical_subset_decision'], nonnull_practical, n, 'practical2method')
    for j, row in enumerate(summary['per_rival']):
        for k, method in enumerate(['P1', 'P2']):
            check_event(row['methods'][method]['Holm38_rejection'], per_method_counts[j][k], n, 'rivalHolm')
            check_event(row['methods'][method]['pointwise_CI95_coverage'], per_method_coverage[j][k], n, 'rivalcoverage')
    mcs_summary = summary['MCS_bootstrap_t_frozen_control']
    check_event(mcs_summary['initial_global_rejection'], mcs_global, n, 'MCSglobal')
    check_event(mcs_summary['all_true_best_retained'], all_best, n, 'MCSallbest')
    check_event(mcs_summary['any_true_best_excluded'], n-all_best, n, 'MCSanybestexcluded')
    check_event(mcs_summary['false_unique_candidate'], false_unique, n, 'MCSfalseunique')


def export(root):
    root = Path(root)
    plan, receipt, overall, decision, records, max_holm_error = load_closed(root)
    output = root/'exports'
    require(not output.exists(), 'New exports directory required; no overwrite')
    case_map = {case['id']: case for case in plan['cases']}
    family_rows, coverage_rows, power_rows, diagnostic_rows = [], [], [], []
    coverage_findings = []
    for summary in overall['cases']:
        case = case_map[summary['id']]
        recount_case(case, summary, records[case['id']])
        row = {'case': case['id'], 'planned_replicates': case['replicates'],
               'completed_replicates': summary['completed_replicates'],
               'known_true_null_rivals': len(summary['true_null_rivals']),
               'global_EPA_null_true': summary['global_EPA_null_true'],
               'P1_unsupported_replicates': summary.get('unsupported_replicates_by_method', {}).get('P1'),
               'P2_unsupported_replicates': summary.get('unsupported_replicates_by_method', {}).get('P2')}
        for key, prefix in [('known_true_null_component_FWER_Holm38', 'new_component_FWER'),
                            ('old_basic_control_known_true_null_FWER_Holm38', 'old_basic_component_FWER'),
                            ('known_true_null_two_method_intersection_event', 'two_method_intersection'),
                            ('known_true_null_directional_practical_subset_event', 'directional_practical_subset')]:
            row.update(scalar_rate(summary.get(key), prefix))
        for method in ['P1', 'P2']:
            row.update(scalar_rate(summary.get('known_true_null_FWER_by_component_within_same_Holm38', {}).get(method),
                                   method+'_component_within_new_Holm38'))
        row.update(scalar_rate(summary.get('simultaneous_all19_by2_pointwiseCI_coverage_diagnostic'),
                               'all19by2_pointwiseCI_joint_coverage_diagnostic'))
        mcs = summary.get('MCS_bootstrap_t_frozen_control', {})
        row['MCS_unsupported_replicates'] = mcs.get('unsupported_replicates')
        for key, prefix in [('initial_global_rejection', 'MCS_global'),
                            ('all_true_best_retained', 'MCS_all_true_best_retained'),
                            ('any_true_best_excluded', 'MCS_any_true_best_excluded'),
                            ('false_unique_candidate', 'MCS_false_unique_candidate')]:
            row.update(scalar_rate(mcs.get(key), prefix))
        family_rows.append(row)
        for rival in summary.get('per_rival', []):
            for method, values in rival['methods'].items():
                ci = values['pointwise_CI95_coverage']
                coverage_row = {'case': case['id'], 'rival': rival['rival'], 'method': method,
                                'true_theta': rival['true_theta'],
                                'bias_against_fixed_longrun_truth': rival['bias_against_fixed_longrun_truth'],
                                'empirical_SD_theta': rival['empirical_SD_theta'],
                                'mean_CI_width': values['mean_CI_width'],
                                'unsupported_intervals': values['unsupported_intervals']}
                for key, prefix in [('raw_rejection', 'raw_rejection'), ('Holm38_rejection', 'Holm38_rejection'),
                                    ('pointwise_CI95_coverage', 'pointwise_coverage')]:
                    coverage_row.update(scalar_rate(values[key], prefix))
                coverage_rows.append(coverage_row)
                if ci['screen'] == 'undercoverage_detected':
                    coverage_findings.append({'case': case['id'], 'rival': rival['rival'], 'method': method,
                                              'events': ci['events'], 'n': ci['replicates'], 'CI95': ci['CI95']})
        if 'all_nonnull_two_method_joint_Holm38_rejection' in summary:
            power = {'case': case['id'], 'nonnull_rivals': sum(abs(b-case['bases'][0]) > 1e-12 for b in case['bases'][1:])}
            for key, prefix in [('all_nonnull_two_method_joint_Holm38_rejection', 'joint_two_method_equality'),
                                ('all_nonnull_two_method_practical_subset_decision', 'joint_two_method_practical')]:
                power.update(scalar_rate(summary[key], prefix))
            power.update(scalar_rate(mcs.get('unique_candidate_set'), 'MCS_unique_candidate'))
            power_rows.append(power)
        diag = summary.get('paired_old_basic_mechanism_diagnostic')
        if diag:
            diagnostic_rows.append({'case': case['id'], **diag})
    interpretation = {
        'kind': 'POST_RESULT_RECOUNT_AND_INTERPRETATION_NOT_PREREGISTRATION',
        'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'campaign_status': receipt['status'], 'complete': overall['complete'],
        'planned_replicates': 1664, 'completed_replicates': receipt['completed_replicates'],
        'actual_crossed_bootstrap_draws': receipt['actual_crossed_bootstrap_draws'],
        'public_anchor_commit': receipt['public_anchor_commit'],
        'raw_log_SHA256': receipt['raw_log_SHA256'],
        'plan_SHA256': receipt['plan_SHA256'], 'source_manifest_SHA256': receipt['source_manifest_SHA256'],
        'all38_new_and_control_Holm_recomputed': True, 'max_Holm_absolute_difference': max_holm_error,
        'major_endpoint_counts_recomputed_from_closed_raw': True,
        'CP95_intervals': 'Copied from frozen driver summaries; individual not simultaneous. No new MC.',
        'driver_FWER_decision': decision,
        'coverage_undercoverage_findings': coverage_findings,
        'coverage_findings_are_multiple_descriptive_screens_not_a_new_multiplicity_corrected_test': True,
        'scope': {
            'estimand': plan['primary_estimand'],
            'family': '2proposedmethods×19rivals Holm38, including real partialnulls',
            'P1': 'New development pairwise crossed bootstrap-t, selected after previous P1 failure',
            'P2': 'Frozen crossed delta-HAC8 t(R−1)/h1 control; unchanged theoretical limits',
            'MCS': 'Frozen global/set diagnostic, no per-rival third p',
            'practical_subset_power': 'Only2method subset with direction and pointwise lowerCIs>delta; not full3way scientificdecision',
            'simultaneous_CI_coverage_claimed': False,
            'heterogeneous_model_temporal_truth': 'Population longrun theta remains0; no finitecalendar recentering',
            'passing_tolerance_screens': 'Does not establish exactnominallevel, marketadmissibility, allpopulationvalidity or thirdmethod'},
        'source_implementation_changed_after_results': False,
        'new_random_draws_by_exporter': 0, 'new_market_fits': 0, 'new_market_predictions': 0,
        'new_market_forwards': 0, 'R3_accredited': False, 'third_primary_method_accredited': False,
        'confirmation_allowed': False, 'AI_assistance_declared': True}
    output.mkdir()
    write_csv(output/'family_and_MCS.csv', family_rows)
    write_csv(output/'pointwise_coverage_and_pairwise.csv', coverage_rows)
    write_csv(output/'two_method_subset_power.csv', power_rows)
    write_csv(output/'paired_mechanism_comparisons.csv', diagnostic_rows)
    write_json(root/'RESULTS_INTERPRETATION.json', interpretation)
    paths = [root/'RESULTS_INTERPRETATION.json']+sorted(output.glob('*.csv'))
    write_json(root/'EXPORT_MANIFEST.json', {
        'kind': 'closed_campaign_tables_and_interpretation_exact_bytes',
        'files': [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                   'sha256': sha(path)} for path in paths],
        'raw_log_SHA256': receipt['raw_log_SHA256'], 'new_random_draws': 0})
    return {'status': 'closed_campaign_recount_and_export_completed',
            'completed_replicates': receipt['completed_replicates'], 'max_Holm_error': max_holm_error,
            'coverage_undercoverage_findings': len(coverage_findings), 'new_random_draws': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(args.execute, 'Explicit --execute required to read a closed result and create exports')
    print(json.dumps(export(args.root), sort_keys=True))
