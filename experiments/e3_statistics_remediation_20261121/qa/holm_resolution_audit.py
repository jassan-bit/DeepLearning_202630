"""Deterministic p-value resolution checks, including a mixed-family counterexample.

AI assistance declared. No simulations or model execution.
"""
from pathlib import Path
import argparse
import json
import math


def holm(values):
    if not values or any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
        raise ValueError('Invalid p-values')
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [None] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def report():
    B = 511
    mixed = [1e-12] * 19 + [1 / (B + 1)] * 19
    adjusted = holm(mixed)
    all_discrete = holm([1 / (B + 1)] * 38)
    assert max(adjusted) == 19 / 512 < .05
    assert min(all_discrete) == 38 / 512 > .05
    return {
        'kind': 'deterministic_Holm_resolution_audit',
        'AI_assistance_declared': True,
        'new_simulations': 0,
        'minimum_bootstrap_raw_p': '1/(B+1)',
        'sufficient_first_rank_resolution_condition': 'm/(B+1) < alpha',
        'condition_is_not_necessary_for_a_mixed_family': True,
        'strict_alpha_05_first_rank_B_minimum': {'m38': 760, 'm57': 1140, 'm63': 1260},
        'mixed_family_counterexample': {
            'B': B, 'analytical_p_values': 19, 'analytical_p_value': 1e-12,
            'bootstrap_p_values': 19, 'bootstrap_p_value': 1 / 512,
            'maximum_Holm_adjusted_p': max(adjusted),
            'all_38_rejected_at_strict_alpha_05': True,
        },
        'all_discrete_family_counterexample': {
            'B': B, 'p_values': 38, 'raw_p': 1 / 512,
            'minimum_Holm_adjusted_p': min(all_discrete),
            'any_rejected_at_strict_alpha_05': False,
        },
        'interpretation':
            'Budget choice must examine achievable p-values by method and Holm rank; insufficient first-rank resolution alone does not establish zero FWER or zero joint power in a mixed family.',
        'calibration_or_power_certificate': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(), indent=2) + '\n')
