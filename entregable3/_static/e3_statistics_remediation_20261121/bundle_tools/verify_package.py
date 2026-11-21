#!/usr/bin/env python3
"""Verify statistical-remediation bytes and bounded claims without executing analyses.

AI assistance declared. This module imports no forecasting/statistical implementation,
draws no samples and treats an honest partial validation as partial evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

BASE = 'd6cf7f1af6800ec1ac85a26347bdde52380e362f'
REQUIRED_COMPONENTS = {
    'calibration/MANIFEST.json', 'retrospective/MANIFEST.json',
    'guide_power/MANIFEST.json', 'bundle_tools/MANIFEST.json',
}
FORBIDDEN_PARTS = {'__pycache__', '.git', 'private_cases', 'private_case_current',
                   'source-checkout', 'wheelhouse', 'book_wheelhouse', 'before'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.npz', '.npy', '.whl', '.pyc', '.pyo',
                      '.onnx', '.safetensors', '.ckpt'}


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def safe_path(value):
    require(isinstance(value, str) and value and '\x00' not in value,
            'empty or invalid member path')
    p = PurePosixPath(value)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == value
            and '\\' not in value, 'unsafe/noncanonical member: ' + value)
    require(not FORBIDDEN_PARTS.intersection(p.parts)
            and p.suffix.lower() not in FORBIDDEN_SUFFIXES,
            'excluded private/model/cache payload: ' + value)
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decode_json(text):
    def unique(pairs):
        d = {}
        for key, value in pairs:
            require(key not in d, 'duplicate JSON key: ' + key)
            d[key] = value
        return d
    def reject(value):
        raise VerificationError('nonfinite JSON: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def read_json(root, relative):
    p = root.joinpath(*safe_path(relative).parts)
    require(p.is_file() and not p.is_symlink(), 'missing/linked JSON: ' + relative)
    value = decode_json(p.read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'JSON object required: ' + relative)
    return value


def actual_members(root):
    found = set()
    for p in root.rglob('*'):
        rel = p.relative_to(root).as_posix()
        require(not p.is_symlink(), 'symlink in immutable packet: ' + rel)
        if p.is_file():
            safe_path(rel)
            found.add(rel)
    return found


def check_entries(root, entries, prefix=''):
    require(isinstance(entries, list) and entries, 'manifest has no payload list')
    names, total = set(), 0
    for e in entries:
        require(isinstance(e, dict), 'invalid payload record')
        rel = e.get('path'); safe_path(rel)
        full = prefix + rel; safe_path(full)
        require(full not in names, 'duplicate member: ' + full)
        require(type(e.get('bytes')) is int and e['bytes'] >= 0,
                'invalid payload size: ' + full)
        require(isinstance(e.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', e['sha256']), 'invalid hash: ' + full)
        p = root.joinpath(*PurePosixPath(full).parts)
        require(p.is_file() and not p.is_symlink(), 'missing/linked member: ' + full)
        require(p.stat().st_size == e['bytes'] and digest(p) == e['sha256'],
                'payload bytes/hash differ: ' + full)
        names.add(full); total += e['bytes']
    return names, total


def unchanged_false(d, keys, label):
    for k in keys:
        require(d.get(k) is False, label + ' unsupported assertion: ' + k)


def zero_fields(d, keys, label):
    for k in keys:
        require(type(d.get(k)) is int and d[k] == 0, label + ' nonzero/absent counter: ' + k)


def check_semantics(root, packet):
    require(packet.get('source_base_commit') == BASE, 'packet base source differs')
    zero_fields(packet, ['new_forecasting_model_fits', 'new_forecasting_model_forward_passes',
                         'new_market_predictions'], 'packet')
    unchanged_false(packet, ['R3_accredited', 'confirmation_enabled', 'human_execution_claimed'], 'packet')
    initial = read_json(root, 'initial_state.json')
    require(initial.get('base_commit') == BASE, 'initial source differs')
    unchanged_false(initial, ['new_model_fits_allowed', 'new_forecasts_allowed', 'R3_accredited'], 'initial policy')
    preserve = initial.get('preserve', {})
    require(isinstance(preserve, dict), 'historical inventory is not an object')
    for k, value in [('additive_jobs', 796), ('V3_fits', 30), ('transfer_trajectories', 51),
                     ('transfer_rows', 20400), ('V4_fits', 20), ('V4_rows', 1210)]:
        require(preserve.get(k) == value, 'historical inventory changed: ' + k)
    require(preserve.get('V3_NO_GO') is True and preserve.get('V4_abandoned') is True,
            'historical scientific stop changed')
    require(initial.get('oral_score_provided_by_author') == 4.5
            and initial.get('observed_defense_claimed') is False,
            'provided oral score misrepresented as observed defense')

    plan = read_json(root, 'calibration/VALIDATION_PLAN.json')
    code = read_json(root, 'calibration/CODE_MANIFEST.json')
    anchor = read_json(root, 'calibration/PUBLIC_ANCHOR.json')
    receipt = read_json(root, 'calibration/validation_fresh/execution_receipt.json')
    summary = read_json(root, 'calibration/validation_fresh/validation_summary.json')
    expected = sum(c['replicates'] for c in plan['cases'])
    require(expected == plan.get('planned_replicates') == 640
            and len(plan['cases']) == 4 and plan['settings']['B'] == 999,
            'bounded validation design differs')
    require(plan.get('validation_seed') != plan.get('old_validation_seed_not_reused'),
            'old validation RNG reused')
    unchanged_false(plan, ['confirmation_allowed', 'R3_accredited'], 'validation plan')
    zero_fields(plan, ['new_market_fits', 'new_market_predictions'], 'validation plan')
    require(anchor.get('status') == 'MCS_bootstrap_t_code_and_plan_published_before_validation'
            and anchor.get('observed_by_root_before_validation') is True
            and re.fullmatch('[0-9a-f]{40}', str(anchor.get('commit', ''))),
            'immutable validation publication anchor absent')
    require(anchor.get('plan_SHA256') == digest(root/'calibration/VALIDATION_PLAN.json')
            and anchor.get('source_manifest_SHA256') == digest(root/'calibration/CODE_MANIFEST.json'),
            'validation publication anchor source hashes differ')
    check_entries(root, code['files'], 'calibration/')
    require(receipt.get('plan_SHA256') == anchor['plan_SHA256']
            and receipt.get('source_manifest_SHA256') == anchor['source_manifest_SHA256']
            and receipt.get('public_anchor_commit') == anchor['commit']
            and receipt.get('source_hashes_still_match') is True,
            'executed source differs from prior anchor')
    n = receipt.get('completed_replicates')
    require(type(n) is int and 0 <= n <= expected, 'invalid validation replication count')
    require(receipt.get('planned_replicates') == expected
            and receipt.get('actual_crossed_bootstrap_draws') == n * plan['settings']['B']
            and receipt.get('old_control_extra_bootstrap_draws') == 0,
            'validation draw accounting differs')
    status = receipt.get('status')
    require(status in {'completed', 'resource_stop', 'error'}, 'validation has not stopped')
    require(isinstance(receipt.get('errors'),list)
            and bool(receipt['errors']) is (status == 'error')
            and (status != 'resource_stop' or n < expected),
            'error/resource-stop validation status contradicts its error log')
    complete = status == 'completed'
    require((not complete or (n == expected and not receipt.get('errors')))
            and summary.get('complete') is complete and summary.get('status') == status,
            'partial/error validation mislabeled complete')
    require(summary.get('public_anchor_commit') == anchor['commit']
            and summary.get('source_manifest_SHA256') == anchor['source_manifest_SHA256'],
            'validation summary is not source-bound')
    zero_fields(receipt, ['new_market_fits', 'new_market_predictions'], 'validation')
    zero_fields(summary, ['new_market_fits', 'new_market_predictions'], 'validation summary')
    unchanged_false(receipt, ['R3_accredited', 'human_execution_claimed',
                              'source_implementation_changed_after_results'], 'validation')
    unchanged_false(summary, ['three_primary_pairwise_tests_accredited',
                             'complete_three_way_power_computed',
                             'final_market_calibration_certified', 'final_market_power_certified',
                             'R3_accredited'], 'validation summary')
    require(summary.get('bootstrap_variants_are_not_three_distinct_primary_tests') is True,
            'bootstrap variants counted as distinct primary procedures')
    cases = summary.get('cases', [])
    require(isinstance(cases, list) and all(isinstance(c,dict) for c in cases),
            'validation cases must be objects')
    planned_cases = {c['id']: c['replicates'] for c in plan['cases']}
    ids = [c.get('id') for c in cases]
    require(len(ids) == len(set(ids)) and set(ids) <= set(planned_cases), 'unknown/duplicate case')
    require(sum(c.get('completed_replicates', -1) for c in cases) == n,
            'case replication accounting differs')
    for c in cases:
        require(c.get('planned_replicates') == planned_cases[c['id']]
                and 0 <= c['completed_replicates'] <= c['planned_replicates'], 'invalid case completion')
    require(not complete or set(ids) == set(planned_cases), 'completed validation omits a case')
    raw_counts = {name:[] for name in planned_cases}; seen = set(); errors = 0
    raw_path = root/'calibration/validation_fresh/per_replicate.jsonl.gz'
    with gzip.open(raw_path, 'rt', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            require(line_number <= expected + 1 and len(line) <= 2_000_000,
                    'unexpectedly large controlled log')
            item = decode_json(line)
            require(isinstance(item,dict), 'controlled replication is not an object')
            name = item.get('case'); rep = item.get('replicate')
            require(name in planned_cases and type(rep) is int
                    and 0 <= rep < planned_cases[name]
                    and (name,rep) not in seen
                    and item.get('stage') == 'fresh_controlled_validation',
                    'unknown, duplicate or invalid controlled replication')
            seen.add((name,rep))
            if 'error' in item:
                errors += 1
                continue
            require(re.fullmatch('[0-9a-f]{64}', str(item.get('q_SHA256',''))),
                    'controlled input fingerprint absent')
            result = item.get('result', {})
            require(isinstance(result,dict), 'controlled result is not an object')
            require(result.get('shape') == [12,128,20]
                    and result.get('pairwise_methods') == 2
                    and len(result.get('pairwise',[])) == 19,
                    'controlled replicate design differs')
            unchanged_false(result, ['third_primary_method_accredited',
                                     'confirmed_valid_for_market', 'R3_accredited'], 'controlled replicate')
            zero_fields(result, ['new_market_fits', 'new_market_predictions'], 'controlled replicate')
            raw_counts[name].append(rep)
    require(sum(map(len,raw_counts.values())) == n and errors == len(receipt.get('errors',[])),
            'raw controlled log accounting differs from receipt')
    for c in cases:
        require(sorted(raw_counts[c['id']]) == list(range(c['completed_replicates'])),
                'controlled replications are missing or repeated')

    retro = read_json(root, 'retrospective/results/analysis_summary.json')
    rr = read_json(root, 'retrospective/results/execution_receipt.json')
    require(rr.get('status') == 'completed_retrospective_adequacy_not_confirmation',
            'retrospective analysis is not complete')
    zero_fields(rr, ['new_forecasting_model_fits', 'new_forecasting_model_forward_passes',
                     'new_market_predictions', 'new_bootstrap_draws'], 'retrospective receipt')
    unchanged_false(rr, ['R3_accredited', 'confirmation_enabled', 'human_execution_claimed',
                        'primary_cross_seed_R10_guard_relaxed'], 'retrospective receipt')
    require(rr.get('frozen_inputs_after_execution_unchanged') is True,
            'retrospective inputs changed during execution')
    for k, value in [('trajectories',22), ('recipes',6), ('prediction_points',1210),
                     ('diagnostic_family_slots',88), ('pointwise_asymptotic_mean_error_intervals',44),
                     ('new_auxiliary_OLS_regressions',44), ('primary_minimum_seeds_required',10),
                     ('neural_seed_count_observed',5)]:
        require(retro.get(k) == value, 'retrospective count/scope differs: ' + k)
    require(retro.get('computed_diagnostic_p_values', -1)
            + retro.get('unsupported_reserved_slots', -1) == 88,
            'invalid diagnostic slots silently dropped')
    unchanged_false(retro, ['primary_cross_seed_inference_run', 'formal_overfit_adequacy_pass',
                           'V4_predictive_interval_coverage_or_PIT_available',
                           'V4_calibrator_invented', 'simultaneous_coverage_claimed', 'R3_accredited'], 'retrospective')
    require(retro.get('V3_original_NO_GO_preserved') is True
            and retro.get('V4_original_abandon_decision_preserved') is True,
            'retrospective analysis changed the prior model verdict')
    zero_fields(retro, ['new_forecaster_fits', 'new_forecaster_forward_passes',
                        'new_bootstrap_draws'], 'retrospective')

    power = read_json(root, 'guide_power/EXECUTED_POWER_INPUT_AUDIT.json')
    for k, value in [('saved_scenarios_reconciled',486),
                     ('older_distinct_planning_sources_reverified',71),
                     ('old_V3_available_development_variances',3),
                     ('old_V3_missing_development_variances',16),
                     ('old_V3_future_cells_missing',76), ('V4_future_cells_missing',84),
                     ('new_V4_D00_descriptive_pairs',5), ('new_V4_descriptive_covariance_entries',25),
                     ('admitted_future_panels',0)]:
        require(power.get(k) == value, 'power audit inputs/scope differ: ' + k)
    require(power.get('final_R') is None and power.get('final_T') is None,
            'a final design was invented')
    zero_fields(power, ['new_model_fits', 'new_model_predictions',
                       'new_statistical_resamples', 'new_market_p_values'], 'power audit')
    unchanged_false(power, ['new_V4_refs_are_future_inputs', 'planning340fits_authorized',
                           'full_battery_power_established', 'guide_joint90_required',
                           'zero_imputation_used', 'confirmation_enabled', 'R3_accredited'], 'power audit')
    return {'controlled_validation_status':status, 'new_validation_replicates':n,
            'new_validation_bootstrap_draws':n*plan['settings']['B'],
            'retrospective_diagnostic_slots':88, 'retrospective_auxiliary_OLS_regressions':44,
            'new_forecasting_model_fits':0, 'new_market_predictions':0,
            'R3_accredited':False, 'confirmation_enabled':False,
            'analysis_or_sampling_executed_by_verifier':False}


def verify_package(packet):
    root = Path(packet).resolve()
    require(root.is_dir(), 'packet directory missing')
    actual = actual_members(root)  # Reject linked ancestors before reading/hashing any payload.
    manifest = read_json(root, 'PACKET_MANIFEST.json')
    require(manifest.get('schema_version') == 1, 'unsupported packet schema')
    expected, total = check_entries(root, manifest.get('files'))
    require('PACKET_MANIFEST.json' not in expected, 'packet manifest must not hash itself')
    require(actual == expected | {'PACKET_MANIFEST.json'}, 'packet membership differs')
    components = manifest.get('components', [])
    require(isinstance(components, list) and all(isinstance(c,dict) for c in components),
            'component records must be objects')
    names = set()
    for c in components:
        rel = c.get('manifest'); safe_path(rel)
        require(rel not in names and rel in expected, 'duplicate/absent component manifest')
        require(c.get('sha256') == digest(root/rel), 'component manifest hash differs: ' + rel)
        cm = read_json(root, rel); prefix = str(PurePosixPath(rel).parent) + '/'
        members, _ = check_entries(root, cm.get('files'), prefix)
        actual = {p for p in expected if p.startswith(prefix)}
        require(actual == members | {rel}, 'component membership differs: ' + rel)
        names.add(rel)
    require(REQUIRED_COMPONENTS <= names, 'required component manifest missing')
    scope = check_semantics(root, manifest)
    return {'status':'PASS', 'files':len(expected)+1, 'payload_bytes':total,
            'components_verified':len(names), 'packet_manifest_sha256':digest(root/'PACKET_MANIFEST.json'),
            **scope}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); may_write = True
    try:
        if a.output:
            may_write = not a.output.resolve().is_relative_to(a.packet.resolve())
            require(may_write, 'report must be outside immutable packet')
        result = verify_package(a.packet); code = 0
    except (VerificationError, OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        result = {'status':'FAIL', 'error':str(exc), 'analysis_or_sampling_executed_by_verifier':False}; code = 1
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if a.output and may_write:
        a.output.write_text(text, encoding='utf-8')
    print(text, end=''); return code


if __name__ == '__main__':
    sys.exit(main())
