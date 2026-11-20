"""Exact finite-law checks only: no optimizer, market data, model inference or fit.

These are checks of standard probability identities, not evidence of novelty.
The positive result compares nonlinear expectation to a second-moment plug-in;
it does not compare the actual additive neural reference to V3.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random

C = 6.0 / 49.0


def expectation(values, masses):
    assert len(values) == len(masses) and all(p >= 0 for p in masses)
    assert abs(sum(masses) - 1) < 1e-12
    return sum(v * p for v, p in zip(values, masses))


def transformed_mean(A, atoms, masses):
    return expectation([math.sqrt(A + C * u * u) for u in atoms], masses)


def plugin(A, atoms, masses):
    return math.sqrt(A + C * expectation([u * u for u in atoms], masses))


def crps(atoms, masses, observation):
    first = expectation([abs(u - observation) for u in atoms], masses)
    pair = sum(p * q * abs(u - v)
               for u, p in zip(atoms, masses) for v, q in zip(atoms, masses))
    return first - .5 * pair


def conditional_squared_risk(prediction, A, true_atoms, true_masses):
    return expectation([(prediction - math.sqrt(A + C * u * u)) ** 2
                        for u in true_atoms], true_masses)


def main():
    rng = random.Random(2026112001)
    rows = []
    for i in range(100):
        A = 10 ** rng.uniform(-3, 1)
        atoms = sorted(10 ** rng.uniform(-2, 1) for _ in range(3))
        raw = [rng.uniform(.1, 1) for _ in atoms]
        masses = [p / sum(raw) for p in raw]
        mean = transformed_mean(A, atoms, masses)
        plug = plugin(A, atoms, masses)
        V = [u * u for u in atoms]
        EV = expectation(V, masses)
        variance = expectation([(v - EV) ** 2 for v in V], masses)
        lower = C * C * variance / (8 * (A + C * max(V)) ** 1.5)
        upper = C * C * variance / (8 * (A + C * min(V)) ** 1.5)
        gap = plug - mean
        tolerance = 5e-13 * max(1, plug, upper)
        assert lower - tolerance <= gap <= upper + tolerance
        risk_gap = conditional_squared_risk(plug, A, atoms, masses) - conditional_squared_risk(mean, A, atoms, masses)
        assert abs(risk_gap - gap * gap) < 5e-12 * max(1, abs(risk_gap))
        rows.append({'case': i, 'A': A, 'gap': gap,
                     'curvature_lower': lower, 'curvature_upper': upper,
                     'plugin_minus_exact_squared_risk': risk_gap})

    # Same conditional E[U^2] does not determine E[sqrt(A+c U^2)].
    A = .04
    law_one = ([math.sqrt(.2), math.sqrt(1.8)], [.5, .5])
    law_two = ([math.sqrt(.8), math.sqrt(1.2)], [.5, .5])
    means = [transformed_mean(A, *law) for law in (law_one, law_two)]
    plugins = [plugin(A, *law) for law in (law_one, law_two)]
    assert abs(plugins[0] - plugins[1]) < 1e-14 and abs(means[0] - means[1]) > 1e-3

    # The additive scalar head can express the same point forecast. V3 does not
    # establish a richer point-prediction function class by this example.
    point = means[1]
    normal_scale = 1
    epsilon = 1e-4
    residual = (point - math.sqrt(A)) / normal_scale - epsilon
    assert residual > 0
    logit = math.log(math.expm1(residual))
    recovered = math.sqrt(A) + normal_scale * (epsilon + math.log1p(math.exp(logit)))
    assert abs(point - recovered) < 1e-14

    # Restricted/misspecified approximation: an auxiliary proper score can
    # reverse point-risk ordering among finite-law candidates. This is not an
    # optimizer experiment and does not claim these are global V3 optima.
    A = 1.0
    true_atoms, true_masses = [.01, 1, 8, 12], [.35, .2, .25, .2]
    target_mean = transformed_mean(A, true_atoms, true_masses)
    center = math.sqrt((target_mean * target_mean - A) / C)
    approximate_masses = [.3, .4, .3]
    approximations = []
    for shift in (0, .01):
        # Spacing 2e-4 lies strictly inside the finite-logit atomic parametrization.
        atoms = [center - shift - .0002, center - shift, center - shift + .0002]
        prediction = transformed_mean(A, atoms, approximate_masses)
        mse = conditional_squared_risk(prediction, A, true_atoms, true_masses)
        score = expectation([crps(atoms, approximate_masses, u) for u in true_atoms], true_masses)
        approximations.append({'shift': shift, 'atoms': atoms,
                               'prediction': prediction, 'squared_risk': mse,
                               'CRPS_U_risk': score, 'joint_risk_lambda_0_1': mse + .1 * score})
    assert approximations[0]['squared_risk'] < approximations[1]['squared_risk']
    assert approximations[0]['joint_risk_lambda_0_1'] > approximations[1]['joint_risk_lambda_0_1']

    output = {
        'kind': 'finite_law_analytic_checks_only', 'AI_assistance_declared': True,
        'status': 'passed', 'curvature_and_plugin_risk_cases': len(rows),
        'second_moment_identifiability_counterexample': {'A': .04,
            'same_second_moment_plugin': plugins, 'different_exact_means': means},
        'additive_scalar_head_can_represent_exact_point_mean': True,
        'auxiliary_score_point_risk_order_reversal': approximations,
        'comparison_scope': 'Nonlinear expectation versus second-moment plug-in under a correctly specified fixed finite law; NOT a superiority result versus the additive neural model.',
        'standard_Jensen_Taylor_and_squared_risk_identities_not_new_theorems': True,
        'new_real_data_fits': 0, 'new_model_inference': 0,
        'market_data_read': False, 'statistical_resampling': 0,
        'R1_accredited': False, 'R3_accredited': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': rows,
    }
    destination = Path(__file__).with_name('analytic_verification.json')
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
