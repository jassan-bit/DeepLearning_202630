#!/usr/bin/env python3
"""Offline integrity and claim-scope audit; never imports or runs model code."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

SCHEMA = 1
PINNED_COMPONENTS = {
    'development/MANIFEST.json': '7f0e78fca26c5d80a31548dfbf8d704ca2afd8281b21970e35270b661c5131a7',
    'statistics/MANIFEST.json': '864f02293401a93040f9d3256bc50755393ab0d6c3576eaf47f025b072ddc885',
    'data_admission/MANIFEST.json': '361171c29467867d066904459badd7bb824d79920bef78bd1f886d48e9699bfd',
    'human_reproduction/MANIFEST.json': 'a35a5f56d6288ad5640f347b59415f00968277dceb96096fda0dc83db75b0206',
    'human_reproduction/addon_v1_1/MANIFEST.json': '6818420f3efa61eb38b456f2bc2a9da3bf4052e48b9d210c45b18ede886e116a',
}
REQUIRED_COMPONENTS = set(PINNED_COMPONENTS) | {
    'planning/MANIFEST.json', 'container_profile/MANIFEST.json', 'bundle_tools/MANIFEST.json'
}
BASE_COMMIT = 'f5c960eefedb5fc49b313dd795b1ec84e784bd78'
SOURCE20_SHA = '87ff742443a9e7d353663390ec54ec1c44b902c3d9fcded4ccf3c3af797df205'
CPU_ANCHOR = '5ee55723e7d1ce63dbe4a94dca8941cb85a45fee'

class VerificationError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise VerificationError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_path(value):
    require(isinstance(value, str) and value != '', 'empty/non-string member path')
    p = PurePosixPath(value)
    require(not p.is_absolute() and '..' not in p.parts and '.' not in p.parts,
            'unsafe member path: ' + value)
    require(p.as_posix() == value and '\\' not in value, 'noncanonical member path: ' + value)
    return p

def allowed_path(value):
    p = safe_path(value)
    forbidden_parts = {'private_cases', '__pycache__', '.git', 'source-checkout',
                       'book_wheelhouse', 'docker_cli_config', 'unpublished_plot_draft'}
    require(not forbidden_parts.intersection(p.parts), 'private/cache directory in packet: ' + value)
    require(p.suffix.lower() not in {'.npz', '.pt', '.pth', '.whl', '.pyc', '.pyo'},
            'excluded raw/weight/wheel payload: ' + value)
    return p

def read_json(root, relative):
    path = root.joinpath(*safe_path(relative).parts)
    require(path.is_file() and not path.is_symlink(), 'missing/linked JSON: ' + relative)
    def reject_constant(value):
        raise VerificationError('nonfinite JSON constant: ' + value)
    def unique_object(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key: ' + key)
            out[key] = value
        return out
    try:
        return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant,
                          object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError('invalid JSON: ' + relative) from exc

def actual_members(root):
    found = set()
    for p in root.rglob('*'):
        relative = p.relative_to(root).as_posix()
        require(not p.is_symlink(), 'symlink in packet: ' + relative)
        if p.is_file():
            allowed_path(relative)
            found.add(relative)
    return found

def verify_entries(root, entries, *, prefix=''):
    require(isinstance(entries, list) and len(entries) > 0, 'file list absent/empty')
    expected = set()
    total = 0
    for entry in entries:
        require(isinstance(entry, dict), 'invalid file entry')
        relative = entry.get('path')
        safe_path(relative)
        full = prefix + relative
        allowed_path(full)
        require(full not in expected, 'duplicate manifest member: ' + full)
        expected.add(full)
        size = entry.get('bytes')
        sha = entry.get('sha256')
        require(type(size) is int and size >= 0, 'invalid bytes: ' + full)
        require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'invalid SHA256: ' + full)
        path = root.joinpath(*PurePosixPath(full).parts)
        require(path.is_file() and not path.is_symlink(), 'missing/linked member: ' + full)
        require(path.stat().st_size == size, 'byte count mismatch: ' + full)
        require(digest(path) == sha, 'SHA256 mismatch: ' + full)
        total += size
    return expected, total

def check_semantics(root):
    dev = read_json(root, 'development/CURRENT_DEVELOPMENT_DECISION.json')
    require(dev.get('decision') == 'ABANDON_CURRENT_V4_RECIPE', 'V4 stop decision changed')
    for key, wanted in [('actual_new_optimizer_fits',20), ('actual_new_assessment_predictions',1210),
                        ('trajectories',22), ('recipes',6), ('preserved_additive_jobs',796),
                        ('preserved_V3_fits',30), ('preserved_transfer_rows',20400),
                        ('new_designs',1), ('paired_neural_seeds',5)]:
        require(dev.get(key) == wanted, 'development count/scope mismatch: ' + key)
    for key, wanted in [('original_V3_NO_GO_preserved',True), ('no_control_promoted',True),
                        ('R1_inherited',False), ('R3_accredited',False), ('V5_started',False),
                        ('confirmation_allowed',False), ('independent_human_execution_claimed',False),
                        ('private_checkpoints_or_NPZ_payload_published',False)]:
        require(dev.get(key) is wanted, 'development claim changed: ' + key)
    require(dev.get('CPU_source_manifest_sha256') == SOURCE20_SHA, 'pre-fit source20 pin changed')
    require(dev.get('CPU_plan_public_anchor_commit') == CPU_ANCHOR, 'pre-fit public anchor changed')
    require(math.isclose(dev.get('percent_candidate_worse', math.nan), 1.4432610846934824,
                         rel_tol=1e-12), 'adverse V4 result changed')
    result = read_json(root, 'development/cpu_pilot_results/pilot_result.json')
    require(result.get('new_real_data_fits') == 20 and result.get('new_assessment_predictions') == 1210
            and result.get('new_assessment_trajectories') == 22, 'exported actual fit/prediction counts changed')
    require(result.get('new_confirmation_predictions') == 0 and result.get('old796_old30_old20400_mutated') is False,
            'historical preservation/confirmation scope changed')
    csv_path = root / 'development/cpu_pilot_results/assessment_predictions.csv'
    trajectories = set()
    recipes = set()
    row_keys = set()
    with csv_path.open(encoding='utf-8', newline='') as handle:
        rows = csv.DictReader(handle)
        require(rows.fieldnames == ['name','seed','date','target','prediction'], 'prediction CSV schema changed')
        for row in rows:
            key = (row['name'], row['seed'], row['date'])
            require(key not in row_keys, 'duplicate prediction row')
            row_keys.add(key)
            trajectories.add(key[:2]); recipes.add(key[0])
            require(all(math.isfinite(float(row[c])) for c in ['target','prediction']), 'nonfinite prediction/target')
    require(len(row_keys) == 1210 and len(trajectories) == 22 and len(recipes) == 6,
            'saved prediction rows/trajectories/recipes mismatch')
    stat = read_json(root, 'statistics/VALIDATION_DECISION.json')
    require(stat.get('controlled_DGP_replicates_completed') == 1024 and
            stat.get('controlled_bootstrap_draws_completed') == 2096128, 'controlled-run counts changed')
    require(stat.get('method_configuration_approved_for_confirmation') is False and
            stat.get('R3_accredited') is False, 'negative controlled-validation decision changed')
    mcs = stat.get('MCS_implemented_extension_EPA_failure', {})
    require(mcs.get('events') == 27 and mcs.get('replicates') == 128 and
            mcs.get('rate') == 27/128 and mcs.get('nominal') == .1 and
            mcs.get('exploratory_tolerance_screen_passed') is False, 'MCS calibration failure changed')
    admission = read_json(root, 'data_admission/MANIFEST.json')
    require(admission.get('datasets_admitted') == 0 and admission.get('R3_accredited') is False,
            'unadmitted proposals relabelled as admitted')
    panels = read_json(root, 'data_admission/concrete_panels.DRAFT.json')
    require(len(panels.get('panels', [])) == 4 and all(p.get('admitted_unexposed') is False
            for p in panels['panels']), 'future panel scope changed')
    human = read_json(root, 'container_profile/EXECUTION_RECEIPT.json')
    require(human.get('status') == 'book_additive19x4_and_V3_software_passed', 'container execution status changed')
    require(human.get('human_execution_claimed') is False and human.get('human_attestation_still_required') is True
            and human.get('R2_or_R3_accredited_by_automation') is False, 'automation relabelled as human/R2/R3')
    require(human.get('source_commit') == BASE_COMMIT, 'human-case source pin changed')
    appendix = human.get('generated_appendix_HTML_verification', {})
    require(appendix.get('rendered_V3_rows') == 7 and appendix.get('rendered_additive_rows') == 76,
            'regenerated principal-table scope changed')
    disposition = read_json(root, 'planning/POST_PILOT_DISPOSITION.json')
    require(disposition.get('current_V4_recipe_abandoned') is True and
            disposition.get('full_family_planning_for_current_V4_authorized') is False and
            disposition.get('confirmation_protocol_registered') is False and
            disposition.get('confirmation_Colab_generated') is False, 'abandoned recipe reactivated without new plan')
    return {'actual_new_neural_development_fits':20, 'new_development_predictions':1210,
            'new_confirmation_predictions':0, 'V4_decision':'ABANDON_CURRENT_V4_RECIPE',
            'controlled_DGP_replicates':1024, 'MCS_EPA_failure_rate':27/128,
            'prospective_datasets_admitted':0, 'human_execution_claimed':False,
            'human_source_commit':BASE_COMMIT, 'R3_accredited':False}

def verify_packet(root):
    root = Path(root).resolve()
    require(root.is_dir(), 'packet directory missing')
    manifest = read_json(root, 'PACKET_MANIFEST.json')
    require(manifest.get('schema_version') == SCHEMA, 'unsupported packet schema')
    members, total = verify_entries(root, manifest.get('files'))
    require('PACKET_MANIFEST.json' not in members, 'packet manifest must not hash itself')
    actual = actual_members(root)
    require(actual == members | {'PACKET_MANIFEST.json'},
            'packet inventory mismatch missing=' + str(sorted(members-actual)) +
            ' extra=' + str(sorted(actual-members-{'PACKET_MANIFEST.json'})))
    component_refs = manifest.get('components', [])
    require(isinstance(component_refs, list), 'components must be a list')
    component_names = [x.get('manifest') for x in component_refs]
    require(len(component_names) == len(set(component_names)), 'duplicate component declaration')
    require(REQUIRED_COMPONENTS.issubset(set(component_names)), 'required component manifests absent')
    reports = []
    for ref in component_refs:
        relative = ref['manifest']
        safe_path(relative)
        require(relative in members, 'component manifest outside root inventory')
        sha = digest(root / relative)
        require(sha == ref.get('sha256'), 'component reference SHA mismatch: ' + relative)
        if relative in PINNED_COMPONENTS:
            require(sha == PINNED_COMPONENTS[relative], 'frozen component pin changed: ' + relative)
        component = read_json(root, relative)
        prefix = PurePosixPath(relative).parent.as_posix() + '/'
        child_members, child_bytes = verify_entries(root, component.get('files'), prefix=prefix)
        own = {x for x in actual if x.startswith(prefix)}
        nested = [str(PurePosixPath(n).parent) + '/' for n in component_names
                  if n != relative and n.startswith(prefix)]
        own = {x for x in own if not any(x.startswith(n) for n in nested)}
        require(own == child_members | {relative}, 'component inventory mismatch: ' + relative)
        reports.append({'manifest':relative, 'sha256':sha, 'payloads':len(child_members), 'bytes':child_bytes})
    scope = check_semantics(root)
    return {'status':'PASS', 'schema_version':SCHEMA, 'packet_payloads':len(members),
            'packet_payload_bytes':total, 'packet_manifest_sha256':digest(root/'PACKET_MANIFEST.json'),
            'components':reports, 'scope_checks':scope,
            'verification_scope':'Offline byte inventory and selected claim-scope guards; no experiment, remote publication, independent human execution or scientific adequacy certified.',
            'AI_assistance_declared':True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, help='Optional report path outside the immutable packet')
    args = parser.parse_args()
    output_allowed = True
    try:
        if args.output:
            packet = args.packet.resolve()
            output_allowed = not args.output.resolve().is_relative_to(packet)
            require(output_allowed, 'report must be outside immutable packet')
        result = verify_packet(args.packet)
        code = 0
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as exc:
        result = {'status':'FAIL', 'error':str(exc), 'experiments_executed':False}
        code = 1
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output and output_allowed:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return code

if __name__ == '__main__':
    sys.exit(main())
