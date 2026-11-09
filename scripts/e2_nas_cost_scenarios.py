"""Arithmetic only: proposed NAS trajectory counts, never training or timing."""
from __future__ import annotations
import argparse
import json


def scenario(models: int, folds: int = 1, finalists: int = 10, phase2_seeds: int = 5,
             reference_already_selected: bool = False) -> dict:
    for name, value in [('models', models), ('folds', folds), ('finalists', finalists), ('phase2_seeds', phase2_seeds)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'models' else 1):
            raise ValueError(name + ' must be an integer in the allowed range; models>=0, others>=1')
    if finalists < 10 or finalists > 1000 or phase2_seeds < 5:
        raise ValueError('Guide requires 10<=k<=1000 and R2>=5')
    if not isinstance(reference_already_selected, bool):
        raise ValueError('reference_already_selected must be boolean')
    if finalists == 1000 and not reference_already_selected:
        raise ValueError('The reference is necessarily among all 1000 designs')
    unique_finalists = finalists + (0 if reference_already_selected else 1)
    return {'status': 'scenario_only_not_executed', 'models_subject_to_NAS': models,
            'folds_or_origins_repeating_complete_search': folds,
            'architectures_per_model': 5**3 * 2**3, 'phase1_seeds': 1,
            'selected_top_k': finalists, 'reference_already_selected': reference_already_selected,
            'unique_finalists_including_reference': unique_finalists, 'phase2_seeds': phase2_seeds,
            'short_trajectories': models * folds * 1000,
            'full_trajectories': models * folds * unique_finalists * phase2_seeds,
            'measured_seconds': None, 'estimated_seconds': None,
            'includes_prior_HPO_final_refits_controls_or_ablation': False,
            'trajectory_unit': 'one complete pipeline; AE phases and inner/refit costs must be included in its measured time, not treated as free',
            'time_expression': 'm * F * (1000 * t_short + (k + c) * R2 * t_full)',
            'reference_indicator_c': int(not reference_already_selected)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=int, required=True)
    p.add_argument('--folds', type=int, default=1)
    p.add_argument('--finalists', type=int, default=10)
    p.add_argument('--phase2-seeds', type=int, default=5)
    p.add_argument('--reference-already-selected', action='store_true')
    a = p.parse_args()
    print(json.dumps(scenario(a.models, a.folds, a.finalists, a.phase2_seeds,
                              a.reference_already_selected), indent=2))

if __name__ == '__main__':
    main()
