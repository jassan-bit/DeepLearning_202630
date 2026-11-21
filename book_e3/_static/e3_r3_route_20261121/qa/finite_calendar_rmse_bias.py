"""Deterministic Gaussian-integral audit, with AI assistance.

This computes exact finite-calendar expectations by numerical quadrature.
No RNG, controlled Monte Carlo, model fits, predictions, or bootstrap draws.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy import integrate, special


def expected_root_mean_square_gaussian(T, phi):
    covariance = phi ** np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(eigenvalues > 0):
        raise ValueError('Positive definite stationary AR covariance required')

    def integrand(u):
        if u == 0:
            return float(eigenvalues.sum() / T)
        log_laplace = -.5 * np.log1p(2 * u * u * eigenvalues / T).sum()
        return float(-np.expm1(log_laplace) / (u * u))

    value, error = integrate.quad(integrand, 0, np.inf,
                                  epsabs=1e-11, epsrel=1e-11, limit=300)
    return {'T': T, 'phi': phi, 'long_run_population_RMSE': 1.,
            'finite_calendar_expected_sample_RMSE': value / math.sqrt(math.pi),
            'quadrature_error_estimate': error / math.sqrt(math.pi)}


def report():
    T = 128
    cases = [expected_root_mean_square_gaussian(T, phi)
             for phi in (0., .35, .85)]
    exact_iid = math.sqrt(2 / T) * math.exp(
        special.gammaln((T + 1) / 2) - special.gammaln(T / 2))
    assert abs(cases[0]['finite_calendar_expected_sample_RMSE'] - exact_iid) < 1e-11
    assert all(cases[k]['finite_calendar_expected_sample_RMSE'] >
               cases[k + 1]['finite_calendar_expected_sample_RMSE']
               for k in range(len(cases) - 1))
    return {
        'kind': 'deterministic_gaussian_quadrature_not_new_MC_validation',
        'AI_assistance_declared': True,
        'question': 'Equal population RMSE under heterogeneous temporal dependence',
        'population': 'Stationary ergodic unit-variance Gaussian AR(1), abs(phi)<1',
        'estimand': 'sqrt(E_time Z_t^2)=1, or its mean over independent seeds',
        'estimator': 'sqrt(mean over T dates of Z_t^2), then mean over seeds',
        'derivation': [
            'C has eigenvalues lambda_i>0 and trace T; sample MSE has law sum_i lambda_i ChiSquare1_i/T.',
            'E sqrt(X) = integral_0^infinity (1-E exp(-u^2 X))/u^2 du / sqrt(pi).',
            'Gaussian Laplace transform is product_i (1+2*u^2*lambda_i/T)^(-1/2).',
            'The expectation is symmetric concave in lambda; it is maximal at equal eigenvalues (IID).',
            'Nonidentity covariance gives a strict Jensen gap relative to IID despite equal population RMSE.',
        ],
        'cases': cases,
        'iid_gamma_reference': exact_iid,
        'maximum_iid_quadrature_reference_error': abs(
            cases[0]['finite_calendar_expected_sample_RMSE'] - exact_iid),
        'iid_minus_phi_085_bias_difference': cases[0]['finite_calendar_expected_sample_RMSE'] -
                                            cases[2]['finite_calendar_expected_sample_RMSE'],
        'scope': [
            'This is a deterministic estimator-bias example, not a test size or coverage calculation.',
            'It does not negate the previous shared-marginal EPA null or excuse its adverse results.',
            'Independent seed averaging preserves these expectations; increasing seeds alone does not remove finite-T bias.',
            'Temporal block resampling, studentization, or Holm do not by themselves prove removal of this bias.',
            'A future controlled plan should test true population equality with heterogeneous dependence or explicitly restrict its claims.',
        ],
        'new_MC_replicates': 0, 'new_random_draws': 0,
        'new_model_fits': 0, 'new_market_predictions': 0,
        'new_bootstrap_draws': 0, 'R3_accredited': False,
    }


if __name__ == '__main__':
    root = Path(__file__).parent
    output = root / 'FINITE_CALENDAR_RMSE_BIAS_AUDIT.json'
    if output.exists():
        raise SystemExit('Refusing to overwrite existing deterministic audit')
    value = report()
    value['source_SHA256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'path': str(output),
                      'SHA256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'cases': value['cases']}, indent=2))
