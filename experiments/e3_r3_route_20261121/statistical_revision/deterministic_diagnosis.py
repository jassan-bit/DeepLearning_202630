"""Read-only old-log diagnosis plus exact algebra; zero random sampling."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats
from crossed_rmse_v2_frozen import functional_and_covariance, pair_scales


def main():
    root = Path(__file__).resolve().parent
    old = Path('/workspace/e3_statistics_remediation_20261121/calibration/validation_fresh/per_replicate.jsonl.gz')
    cases = collections.defaultdict(list)
    with gzip.open(old, 'rt') as handle:
        for line in handle:
            item = json.loads(line)
            cases[item['case']].append(item)
    report = {'kind': 'DETERMINISTIC_REANALYSIS_OF_PREVIOUS_RESULTS_NOT_NEW_VALIDATION',
              'old_log_SHA256': hashlib.sha256(old.read_bytes()).hexdigest(),
              'new_MC_replicates': 0, 'new_random_draws': 0, 'new_market_fits': 0,
              'new_market_predictions': 0, 'R3_accredited': False, 'cases': {}}
    for name, records in cases.items():
        pp = [r['result']['pairwise'] for r in records]
        th = np.array([[x['theta_hat'] for x in rows] for rows in pp])
        sv = np.array([[x['seed_variance'] for x in rows] for rows in pp])
        tv = np.array([[x['time_variance'] for x in rows] for rows in pp])
        p1 = np.array([[x['P1_Holm'] for x in rows] for rows in pp])
        p2 = np.array([[x['P2_Holm'] for x in rows] for rows in pp])
        share = np.divide(tv, sv+tv, out=np.zeros_like(tv), where=(sv+tv) > 0)
        report['cases'][name] = {
            'old_replicates': len(records), 'any_P1_Holm38': int(np.any(p1 < .05, axis=1).sum()),
            'any_P2_Holm38': int(np.any(p2 < .05, axis=1).sum()),
            'any_component_Holm38': int(np.any((p1 < .05) | (p2 < .05), axis=1).sum()),
            'some_rival_both_methods_Holm38': int(np.any((p1 < .05) & (p2 < .05), axis=1).sum()),
            'time_variance_share_median': float(np.median(share)),
            'median_across_rivals_sqrt_mean_plugin_variance_over_empirical_SD':
                float(np.median(np.sqrt((sv+tv).mean(axis=0))/th.std(axis=0, ddof=1))),
            'P2_rejected_component_time_shares': share[p2 < .05].tolist() if name.startswith('EPA') else None}
    R = 12
    amplitude = np.column_stack([1+np.arange(R)/30, 1.1+np.arange(R)**2/300])
    q = np.repeat(amplitude[:, None, :]**2, 32, axis=1)
    risk, covariance, seed, temporal = functional_and_covariance(q[None], 8)
    d = amplitude[:, 1]-amplitude[:, 0]
    report['minimal_seed_only_identity'] = {
        'R': R, 'T': 32, 'theta_from_functional': float(risk[0, 1]-risk[0, 0]),
        'paired_mean': float(d.mean()), 'crossed_SE': float(pair_scales(covariance)[0, 1, 0]),
        'paired_mean_SE': float(d.std(ddof=1)/np.sqrt(R)),
        'time_variance': float(pair_scales(temporal)[0, 1, 0]**2),
        'exact_conditional_basic_bootstrap_mean_variance_over_sample_mean_SE2': (R-1)/R,
        'explanation': 'For iid seed resampling, Var_star(meanDstar)=(R−1)/R times sD²/R. This identifies finite-R scale contraction, not the full cause of old FWER.'}
    z = float(stats.norm.isf(.05/(2*38)))
    report['normal_vs_t_illustration_not_actual_bootstrap_calibration'] = {
        'normal_critical_for_first_Holm38_rank': z,
        't11_critical_for_first_Holm38_rank': float(stats.t.isf(.05/(2*38), df=11)),
        't11_two_sided_tail_at_normal_critical': float(2*stats.t.sf(z, df=11)),
        'interpretation': 'Normal paired seed means give exact t11 after estimating scale, illustrating why unstudentized/fixed-scale extreme tails are fragile atR12. Actual DGP uses lognormal paired differences; no exactt theorem claimed.'}
    report['causal_limit'] = ('Old records support a P1 extreme-tail problem in seed-only despite roughly correct average SE. Strong dependence additionally has averageSE/empSD<1. Bootstrap-t does not prove either general validity or P2 repair; finite-calendar heterogeneity remains.')
    (root/'DETERMINISTIC_DIAGNOSIS.json').write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': 'deterministic_old_log_diagnosis_written', 'new_random_draws': 0,
                      'cases': len(cases)}, sort_keys=True))


if __name__ == '__main__':
    main()
