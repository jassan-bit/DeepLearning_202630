"""Deterministic calculations for the old Gaussian AR controlled DGP.

No random numbers, forecasts, model fits, or new Monte Carlo replicates.
The kernel expressions refer to the temporal squared-Gaussian component;
they are not a formula for the full crossed nonlinear RMSE estimator variance.
AI assistance declared.
"""
from pathlib import Path
import argparse
import json


def kernel_factors(phi, lag=8, stationary_mean_block=12):
    if not -1 < phi < 1 or lag < 0 or stationary_mean_block < 1:
        raise ValueError('Invalid declared AR or kernel setting')
    rho = phi * phi
    full = (1 + rho) / (1 - rho)
    hac = 1 + 2 * sum((1 - k / (lag + 1)) * rho ** k for k in range(1, lag + 1))
    a = (1 - 1 / stationary_mean_block) * rho
    stationary = (1 + a) / (1 - a)
    return {
        'phi': phi, 'squared_process_autocorrelation_factor': rho,
        'infinite_population_LRV_factor': full,
        'Bartlett_lag_8_population_kernel_factor': hac,
        'Bartlett_to_full_LRV_factor_ratio': hac / full,
        'stationary_mean_12_population_kernel_factor': stationary,
        'stationary_to_full_LRV_factor_ratio': stationary / full,
    }


def report():
    # These constants are the already executed historical plan settings.
    return {
        'kind': 'deterministic_mathematical_reaudit_not_new_validation',
        'AI_assistance_declared': True,
        'new_controlled_DGP_replicates': 0, 'new_model_fits': 0,
        'derivation': {
            'process': 'Stationary unit Gaussian AR(1) Z_t; q_t=Z_t^2',
            'Cov(q_0,q_k)': '2*phi^(2*k)',
            'Var(q_0)': 2,
            'true_LRV_over_marginal_variance': '(1+phi^2)/(1-phi^2)',
            'Bartlett_kernel': '1+2*sum(k=1..L, (1-k/(L+1))*phi^(2*k))',
            'stationary_bootstrap_population_kernel':
                '(1+(1-1/block)*phi^2)/(1-(1-1/block)*phi^2)',
        },
        'temporal_component_scope':
            'Population kernel weights before finite-T demeaning and before seed/interaction terms; not total estimator coverage.',
        'historical_settings': {'HAC_lag': 8, 'stationary_mean_block': 12, 'EPA_R': 12, 'EPA_T': 128, 'EPA_M': 20},
        'cases': [kernel_factors(phi) for phi in (.35, .65, .85)],
        'seed_only_empirical_bootstrap_identity': {
            'statement': 'Conditional variance of the empirical bootstrap mean is ((S-1)/S) times sample_covariance(ddof=1)/S.',
            'S': 12, 'variance_factor': 11 / 12,
            'source': 'Multinomial seed weights, not a new simulation',
        },
        'EPA_null': {
            'all_model_marginals_exchangeable': True,
            'finite_T_Jensen_bias_common_across_models': True,
            'equality_null_still_true_at_finite_T': True,
            'bias_does_not_explain_away_original_27_of_128_rejections': True,
        },
        'limits': [
            'The kernel ratio is one identifiable dependence mechanism, not a proven complete causal attribution of MCS inflation.',
            'Standard MCS also uses a common estimated pair scale for observed and centered bootstrap statistics; this feature alone is not an implementation bug.',
            'A bootstrap-t crossed functional modification is a new proposed procedure and needs a fresh fixed validation.',
            'Corrected studentization alone does not establish validity for every dependence regime or high-dimensional finite seed design.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(), indent=2, allow_nan=False) + '\n')
