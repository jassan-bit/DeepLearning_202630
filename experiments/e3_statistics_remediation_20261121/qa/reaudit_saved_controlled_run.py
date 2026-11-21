"""Read the already published Monte Carlo log; no simulation or model execution.

AI assistance declared. These are post-result development diagnostics only.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import statistics


P1 = 'P1_CROSSED_STATIONARY_BOOTSTRAP_RMSE'
P2 = 'P2_CROSSED_DELTA_HAC_HLN_PROPOSED_EXTENSION'
LOG_SHA256 = 'a687d180084b906631a625db371c137754e3f5f66180fabd3e758ab3328b1b7b'


def read_saved(log_path):
    raw = Path(log_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOG_SHA256:
        raise ValueError('Historical log changed')
    groups = {}
    for line in gzip.decompress(raw).decode().splitlines():
        row = json.loads(line)
        groups.setdefault(row['case'], []).append(row)
    if sum(map(len, groups.values())) != 1024 or len(groups) != 11:
        raise ValueError('Historical case inventory differs')
    cases = {}
    for name, rows in groups.items():
        pair = [x['result']['pairwise'] for x in rows]
        R, T, M = rows[0]['result']['shape']
        details = []
        for j in range(M - 1):
            values = [x[j]['theta_hat'] for x in pair]
            empirical_variance = statistics.variance(values)
            delta_variance = statistics.mean(
                x['result']['pairwise'][j]['variance_components']['variance_used']
                / x['result']['HLN_h1_factor'] ** 2 for x in rows)
            truth = rows[0]['analytic_truth'][j]
            details.append({
                'rival': pair[0][j]['rival'],
                'theta_empirical_mean': statistics.mean(values),
                'theta_analytic_longrun': truth,
                'empirical_variance_over_saved_MC': empirical_variance,
                'mean_delta_variance_with_HLN': delta_variance,
                'delta_variance_to_empirical_variance_ratio':
                    delta_variance / empirical_variance if empirical_variance else None,
                'P1_CI95_coverage': statistics.mean(
                    x[j]['methods'][P1]['CI95'][0] <= truth <= x[j]['methods'][P1]['CI95'][1]
                    for x in pair),
                'P2_CI95_coverage': statistics.mean(
                    x[j]['methods'][P2]['CI95'][0] <= truth <= x[j]['methods'][P2]['CI95'][1]
                    for x in pair),
            })
        p = [x['result']['global_MCS']['initial_global_p'] for x in rows]
        cases[name] = {
            'R': R, 'T': T, 'M': M, 'pair_count': M * (M - 1) // 2,
            'saved_replicates': len(rows),
            'MCS_global_rejections_alpha_10': sum(x is not None and x < .1 for x in p),
            'MCS_null_is_equal': all(x == rows[0]['bases'][0] for x in rows[0]['bases']),
            'per_rival': details,
        }
    return {
        'kind': 'read_only_reanalysis_of_original_saved_MC_outputs_no_new_simulations',
        'AI_assistance_declared': True,
        'old_raw_log_SHA256': LOG_SHA256,
        'new_controlled_DGP_replicates': 0, 'new_model_fits': 0,
        'method_selection_is_post_result_development': True,
        'statistical_calibration_approved': False,
        'cases': cases,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = read_saved(args.log)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'saved_replicates_reaudited': 1024,
                      'EPA_equal_R': report['cases']['EPA_equal']['R'],
                      'EPA_equal_rejections': report['cases']['EPA_equal']['MCS_global_rejections_alpha_10'],
                      'new_simulations': 0}))
