"""Read-only, version-aware planning evidence audit. AI assistance declared.

This program cannot fit, forecast, simulate or authorize a campaign. Its optional
normal approximation is a planning diagnostic under an explicitly additive
crossed-effects model, never power of the complete three-procedure decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def positive_integer(x):
    return type(x) is int and x > 0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_file(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    require(isinstance(relative, str) and not p.is_absolute()
            and all(x not in ('', '.', '..') for x in p.parts), 'Unsafe input path')
    unresolved = root / p
    current = root
    for component in p.parts:
        current = current / component
        require(not current.is_symlink(), 'Symlink input forbidden')
    resolved = unresolved.resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(), 'Missing or escaping input')
    return resolved


def bound_json(root, binding):
    require(isinstance(binding, dict), 'Evidence binding required')
    p = safe_file(root, binding.get('path', ''))
    require(isinstance(binding.get('sha256'), str) and digest(p) == binding['sha256'],
            'Bound evidence hash mismatch')
    return json.loads(p.read_text())


def load_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def linearized_budget_diagnostic(*, seed_variance, calendar_lrv, interaction_lrv,
                                 dates, practical_margin, planning_alternative,
                                 multiplicity, alpha=.05, target_power=.9):
    """Crossed-model normal approximation, with the temporal information floor.

    Assumes Var(theta_hat)=Vs/R+Vt/T+Vst/(R*T), known nonnegative
    population components and a two-sided Bonferroni reference. It omits
    estimation error, nonlinear bias, HAC uncertainty and the actual three-test
    intersection, so neither R nor its MDE is a final design recommendation.
    """
    require(all(finite(v) and v >= 0 for v in
                (seed_variance, calendar_lrv, interaction_lrv)), 'Invalid variance components')
    require(sum((seed_variance, calendar_lrv, interaction_lrv)) > 0,
            'Zero variance cannot certify information')
    require(positive_integer(dates) and positive_integer(multiplicity), 'Positive T and family size required')
    require(finite(practical_margin) and practical_margin >= 0
            and finite(planning_alternative) and planning_alternative > practical_margin,
            'The planning alternative must exceed the practical margin')
    require(finite(alpha) and 0 < alpha < .5 and finite(target_power) and .9 <= target_power < 1,
            'Require alpha<.5 and per-contrast planning target>=.90')
    critical = NormalDist().inv_cdf(1 - alpha / (2 * multiplicity))
    target_z = NormalDist().inv_cdf(target_power)
    gap = planning_alternative - practical_margin
    permitted_variance = (gap / (critical + target_z)) ** 2
    temporal_floor = calendar_lrv / dates
    residual = permitted_variance - temporal_floor
    result = {'kind': 'normal_additive_crossed_variance_diagnostic_only',
              'null_boundary_used_for_diagnostic': practical_margin,
              'this_is_stronger_than_guide_equal_performance_null': practical_margin > 0,
              'planning_alternative': planning_alternative, 'alternative_minus_margin': gap,
              'dates': dates, 'multiplicity': multiplicity, 'alpha': alpha,
              'per_contrast_target': target_power, 'temporal_variance_floor': temporal_floor,
              'maximum_variance_for_reference': permitted_variance,
              'guide_final_seed_minimum': 10, 'guide_requires_joint90': False,
              'complete_battery_power_computed': False, 'final_R_selected': False,
              'confirmation_enabled': False}
    if residual <= 0:
        result.update(status='TEMPORAL_INFORMATION_FLOOR_BLOCKS_THIS_REFERENCE',
                      reference_R=None,
                      reason='Increasing seeds alone cannot reduce Vt/T below the required variance.')
    else:
        reference_R = max(10, math.ceil((seed_variance + interaction_lrv / dates) / residual))
        variance_at_R = (seed_variance / reference_R + calendar_lrv / dates
                         + interaction_lrv / (reference_R * dates))
        result.update(status='ILLUSTRATIVE_NORMAL_REFERENCE_NOT_DESIGN', reference_R=reference_R,
                      stronger_margin_MDE_reference=(critical + target_z) * math.sqrt(variance_at_R))
    return result


def assess_contract(config, root):
    """Structural compatibility checks. Even all-true never certify science."""
    require(isinstance(config, dict), 'JSON object required')
    version = config.get('candidate_version')
    family = config.get('family', {})
    require(isinstance(family, dict), 'Family object required')
    rivals = family.get('rivals', [])
    methods = family.get('methods', [])
    panels = config.get('panels', [])
    require(isinstance(rivals, list) and all(isinstance(x, str) and x for x in rivals),
            'Rivals must be nonempty strings')
    require(isinstance(methods, list) and all(isinstance(x, str) and x for x in methods),
            'Methods must be nonempty strings')
    require(isinstance(panels, list) and all(isinstance(x, dict) for x in panels),
            'Panels must be objects')
    r, t = config.get('final_R'), config.get('final_T')
    checks = {
        'candidate_viable_before_planning': bool(version) and config.get('candidate_status') == 'viable_for_planning',
        'explicit_unique_rivals': isinstance(rivals, list) and bool(rivals)
                                 and len(set(rivals)) == len(rivals) and version not in rivals,
        'three_distinct_methods_and_mapping': isinstance(methods, list) and len(methods) == len(set(methods)) == 3
                                              and family.get('contrastive_mapping_justified') is True,
        'four_admitted_planning_panels': isinstance(panels, list) and len(panels) == 4
                                        and len({x.get('panel_id') for x in panels if isinstance(x, dict)}) == 4
                                        and all(isinstance(x, dict) and x.get('admitted_unexposed') is True
                                                and x.get('excluded_from_confirmation') is True for x in panels),
        'planning_seed_pairs_at_least5': positive_integer(config.get('planning_seed_pairs'))
                                        and config['planning_seed_pairs'] >= 5,
        'final_minimum10_and_calendar_fixed': positive_integer(r) and r >= 10
                                             and positive_integer(t) and t >= 32,
        'operational_margins_and_separate_alternatives': bool(panels)
            and all(isinstance(x, dict) and finite(x.get('practical_margin')) and x['practical_margin'] >= 0
                    and finite(x.get('planning_alternative'))
                    and x['planning_alternative'] > x['practical_margin'] for x in panels),
        'computational_budget_bound': isinstance(config.get('budget'), dict)
            and positive_integer(config['budget'].get('max_wall_seconds'))
            and positive_integer(config['budget'].get('max_replicates')),
        'persistence_and_no_refit_policy': config.get('persistence', {}).get('verified_checkpoint_required') is True
            and config.get('persistence', {}).get('skip_verified_complete') is True
            and config.get('persistence', {}).get('overwrite_existing') is False,
    }
    evidence_errors = {}
    receipts = {}
    for key in ('scientific_decision', 'admission', 'paired_covariance', 'calibration', 'power'):
        try:
            receipts[key] = bound_json(root, config.get('evidence_bindings', {}).get(key))
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
            evidence_errors[key] = str(exc)
    checks['evidence_bindings_present_and_verified'] = not evidence_errors
    scientific = receipts.get('scientific_decision', {})
    checks['scientific_receipt_permits_this_planning_version'] = (
        bool(version) and scientific.get('candidate_version') == version
        and scientific.get('planning_allowed') is True
        and scientific.get('current_recipe_abandoned') is False)
    admission = receipts.get('admission', {})
    checks['admission_receipt_matches_four_planning_panels'] = (
        len(panels) == 4 and admission.get('admitted_panel_ids') == [x.get('panel_id') for x in panels]
        and admission.get('datasets_admitted') == 4
        and admission.get('all_planning_partitions_excluded_from_confirmation') is True)
    for key in ('paired_covariance', 'calibration', 'power'):
        x = receipts.get(key, {})
        checks[key + '_same_version_and_estimand'] = (bool(version) and x.get('candidate_version') == version
            and all(bool(config.get(k)) and x.get(k) == config[k] for k in
                    ('estimand', 'units', 'seed_uncertainty', 'calendar_uncertainty')))
    calibration = receipts.get('calibration', {})
    checks['calibration_accepted_for_exact_methods'] = (calibration.get('approved_for_scope') is True
        and calibration.get('methods') == methods and bool(methods)
        and calibration.get('controlled_validation_protocol_sha256')
        == config.get('controlled_validation_protocol_sha256')
        and bool(config.get('controlled_validation_protocol_sha256')))
    covariance = receipts.get('paired_covariance', {})
    expected_cells = {(x['panel_id'], rival) for x in panels if isinstance(x, dict) and 'panel_id' in x for rival in rivals}
    cells = covariance.get('cells', [])
    keys = {(x.get('panel_id'), x.get('rival')) for x in cells if isinstance(x, dict)}
    checks['complete_compatible_seed_calendar_covariance'] = (bool(expected_cells)
        and len(cells) == len(expected_cells) and keys == expected_cells
        and all(isinstance(x, dict) and x.get('scope') == 'prospective_planning'
                and x.get('candidate_version') == version and x.get('paired_seed_pairs', 0) >= 5
                and all(finite(x.get(k)) and x[k] >= 0 for k in
                        ('seed_variance', 'calendar_lrv', 'interaction_lrv'))
                and isinstance(x.get('cross_rival_covariance_sha256'), str)
                and len(x['cross_rival_covariance_sha256']) == 64 for x in cells))
    power = receipts.get('power', {})
    power_rows = power.get('per_contrast', [])
    power_keys = {(x.get('panel_id'), x.get('rival')) for x in power_rows if isinstance(x, dict)}
    checks['full_rule_power_at_least90_with_MC_precision'] = (bool(expected_cells)
        and len(power_rows) == len(expected_cells) and power_keys == expected_cells
        and power.get('selected_R') == r and power.get('selected_T') == t
        and power.get('full_decision_rule_sha256') == config.get('full_decision_rule_sha256')
        and bool(config.get('full_decision_rule_sha256'))
        and power.get('uses_observed_confirmation_results') is False
        and power.get('complete_three_procedure_rule') is True
        and positive_integer(power.get('independent_MC_replicates'))
        and all(finite(x.get('simultaneous_MC_lower')) and .9 <= x['simultaneous_MC_lower'] <= 1
                for x in power_rows))
    checks['joint_power_and_precision_reported_without_joint90_rule'] = (
        finite(power.get('joint_power')) and 0 <= power['joint_power'] <= 1
        and isinstance(power.get('joint_MC_interval'), list) and len(power['joint_MC_interval']) == 2
        and all(finite(x) and 0 <= x <= 1 for x in power['joint_MC_interval'])
        and power['joint_MC_interval'][0] <= power['joint_power'] <= power['joint_MC_interval'][1])
    return {'kind': 'structural_power_input_compatibility_not_scientific_accreditation',
            'status': 'STRUCTURALLY_COMPATIBLE_REQUIRES_SCIENTIFIC_REVIEW' if all(checks.values())
                      else 'BLOCKED_MISSING_OR_INCOMPATIBLE_INPUTS',
            'checks': checks, 'blockers': [k for k, v in checks.items() if not v],
            'evidence_errors': evidence_errors, 'final_R': r, 'final_T': t,
            'expected_panel_rival_cells': len(expected_cells),
            'guide_minimum_per_contrast_power': .9, 'guide_joint90_required': False,
            'power_certified_by_this_program': False, 'confirmation_enabled': False,
            'can_train': False, 'can_simulate': False}


def audit_saved_inputs(repo_root, pins_path):
    root = Path(repo_root).resolve()
    pins = json.loads(Path(pins_path).read_text())
    payload = {}
    for record in pins['files']:
        p = safe_file(root, record['path'])
        require(p.stat().st_size == record['bytes'] and digest(p) == record['sha256'],
                'Source payload differs: ' + record['path'])
        payload[record['alias']] = p
    old = json.loads(payload['audit_486'].read_text())
    old_family = load_csv(payload['old_family'])
    old_future = load_csv(payload['old_future'])
    new_future = load_csv(payload['V4_future'])
    refs = load_csv(payload['V4_references'])
    cov = load_csv(payload['V4_covariance'])
    disposition = json.loads(payload['V4_disposition'].read_text())
    data = json.loads(payload['admission'].read_text())
    old_manifest = json.loads(payload['old_input_manifest'].read_text())
    registry = json.loads(payload['old_registry'].read_text())
    seed_scenarios = load_csv(payload['seed_scenarios'])
    calendar_scenarios = load_csv(payload['calendar_scenarios'])
    source_hashes = {}
    for row in old_manifest['files']:
        p = safe_file(root, row['path'])
        require(p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], 'Old planning source differs')
        source_hashes[row['path']] = row['sha256']
    for relative, expected in registry['inputs'].items():
        name = 'experiments/e3_completion_20261120/protocol_power/' + relative
        require(digest(safe_file(root, name)) == expected, 'Old registry source differs')
        require(name not in source_hashes or source_hashes[name] == expected, 'Conflicting source bindings')
        source_hashes[name] = expected
    require(len(source_hashes) == 71, 'Expected 71 distinct source bindings')
    require(old['seed_scenarios'] == 162 and old['calendar_scenarios'] == 324
            and old['total_scenarios'] == 486, 'Planning scenario audit changed')
    require(len(seed_scenarios) == 162 and len(calendar_scenarios) == 324,
            'Saved scenario row count differs')
    require(len({tuple(x[k] for k in ('rival', 'variance_reference', 'target_name', 'planned_gain'))
                 for x in seed_scenarios}) == 162, 'Duplicate seed-scenario identity')
    require(len({tuple(x[k] for k in ('rival', 'HAC_Bartlett_lags', 'temporal_SD_multiplier',
                                    'target_name', 'planned_gain')) for x in calendar_scenarios}) == 324,
            'Duplicate calendar-scenario identity')
    require(all(x['final_R_selected'] == 'False'
                and x['actual_three_test_power_established'] == 'False' for x in seed_scenarios)
            and all(x['reference_is_recommended_calendar_length'] == 'False'
                    and x['guide_temporal_three_test_power_established'] == 'False' for x in calendar_scenarios),
            'Saved illustrative scenarios were promoted into a sufficient design')
    require(len(old_family) == 19 and sum(x['saved_V3_pilot_variance_available'] == 'True'
                                       for x in old_family) == 3, 'Old rival availability changed')
    require(len(old_future) == 76 and len(new_future) == 84, 'Historical family templates changed')
    old_numeric = ('planning_seed_pairs', 'paired_seed_SD', 'temporal_long_run_variance',
                   'seed_calendar_interaction_variance', 'planning_dates')
    new_numeric = ('paired_seed_variance', 'calendar_LRV', 'seed_time_interaction', 'R_final', 'T_final')
    require(all(all(x[k] == '' for k in old_numeric) for x in old_future)
            and all(all(x[k] == '' for k in new_numeric) for x in new_future),
            'Unmeasured future cells were filled or zero-imputed')
    require(len(refs) == 5 and len(cov) == 25 and all(x['paired_seeds'] == '5' for x in refs),
            'CPU descriptive reference dimensions changed')
    require(all(x['independent_confirmation'] == 'False'
                and x['future_F00_F03_variance_cell_filled'] == 'False' for x in refs),
            'Exposed CPU references were promoted')
    require(disposition['current_V4_recipe_abandoned'] is True
            and disposition['full_family_planning_for_current_V4_authorized'] is False,
            'Abandoned recipe planning disposition changed')
    require(data['datasets_admitted'] == 0, 'Pinned admission count changed')
    covariance = {(x['rival_i'], x['rival_j']): float(x['sample_covariance_of_seed_differences']) for x in cov}
    require(len(covariance) == 25, 'Repeated covariance key')
    for row in refs:
        name = row['rival']
        require(math.isclose(covariance[name, name], float(row['sample_variance_paired_difference']),
                             rel_tol=1e-12, abs_tol=1e-25), 'Paired variance/covariance mismatch')
        for other in refs:
            require(math.isclose(covariance[name, other['rival']], covariance[other['rival'], name],
                                 rel_tol=1e-12, abs_tol=1e-25), 'Asymmetric descriptive covariance')
    return {'kind': 'executed_read_only_versioned_power_input_audit',
            'base_commit': pins['base_commit'], 'bound_component_sources_verified': len(pins['files']),
            'older_distinct_planning_sources_reverified': len(source_hashes),
            'saved_scenarios_reconciled': 486, 'old_V3_main_rivals': 19,
            'scenario_rows_and_unique_keys_reverified': True,
            'old_V3_available_development_variances': 3, 'old_V3_missing_development_variances': 16,
            'old_V3_development_seed_pairs': 3, 'guide_planning_minimum_seed_pairs': 5,
            'old_V3_future_cells_missing': 76, 'V4_future_cells_missing': 84,
            'new_V4_D00_descriptive_pairs': 5, 'new_V4_descriptive_covariance_entries': 25,
            'new_V4_refs_are_future_inputs': False, 'V4_recipe_abandoned': True,
            'planning340fits_authorized': False, 'admitted_future_panels': 0,
            'final_R': None, 'final_T': None, 'full_battery_power_established': False,
            'guide_per_contrast_minimum_power': .9, 'guide_joint_power_must_be_reported': True,
            'guide_joint90_required': False, 'zero_imputation_used': False,
            'new_model_fits': 0, 'new_model_predictions': 0, 'new_statistical_resamples': 0,
            'new_market_p_values': 0, 'confirmation_enabled': False, 'R3_accredited': False,
            'AI_assistance_declared': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--pins', type=Path, default=Path(__file__).with_name('INPUT_BINDINGS.json'))
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'New output required; refuse overwrite')
    report = audit_saved_inputs(args.repo_root, args.pins)
    if args.contract:
        report['structural_contract'] = assess_contract(json.loads(args.contract.read_text()), args.contract.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
