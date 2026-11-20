"""Read-only verification of the dated E3 documentary resolution packet.

Only Python's standard library is used. No archived program, model, notebook,
training request, network resource or scientific resample is executed. A pass
checks file identity and declared scope; it does not establish scientific
confirmation, rights clearance or independent human reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

BASE = '8d03a73d529eb3f9b2e89992b4488cfeb88a99e6'
SCIENCE_PIN = 'd6106bd8913b2c93f7e1a3b75644e082ee8595fabc0ebf683bd7d9399032d7c0'
C_PIN = 'e8ab147d6c6e74d581b8b61504dc64ce6ecbbe79f3c2a2e9a1c344537028d5a7'
FIRST_ARCHIVE_PIN = 'd7a97766af592629efafcc64c386c9aeae39b4ab107b909217d9629b9d8a6d35'
MANIFEST = 'PACKET_MANIFEST.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name
            and not any(ord(c) < 32 for c in name), 'Unsafe member name')
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name
            and all(part not in ('', '.', '..') for part in name.split('/'))
            and not re.match(r'^[A-Za-z]:', name), 'Unsafe member path: ' + name)
    return path


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root, name):
    parts = safe_name(name).parts
    current = root
    for part in parts:
        current = current / part
        require(not current.is_symlink(), 'Symlink member or parent: ' + name)
    require(current.is_file(), 'Missing regular file: ' + name)
    require(stat.S_ISREG(current.stat().st_mode), 'Nonregular member: ' + name)
    return current


def validate_entry(entry):
    require(isinstance(entry, dict), 'Member entry must be an object')
    name = entry.get('path')
    safe_name(name)
    require(type(entry.get('bytes')) is int and entry['bytes'] >= 0, 'Invalid byte count: ' + name)
    require(isinstance(entry.get('sha256'), str)
            and re.fullmatch(r'[0-9a-f]{64}', entry['sha256']), 'Invalid SHA256: ' + name)
    return name


def verify_inventory(root, entries):
    require(root.is_dir() and not root.is_symlink(), 'Packet root must be a real directory')
    require(isinstance(entries, list) and entries, 'Manifest files must be a nonempty list')
    declared = set()
    total = 0
    for entry in entries:
        name = validate_entry(entry)
        require(name != MANIFEST and name not in declared, 'Duplicate/self-declared member: ' + name)
        declared.add(name)
        path = checked_path(root, name)
        require(path.stat().st_size == entry['bytes'], 'Byte count mismatch: ' + name)
        require(sha_file(path) == entry['sha256'], 'SHA256 mismatch: ' + name)
        total += entry['bytes']
    observed = set()
    for path in root.rglob('*'):
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), 'Symlink present: ' + path.relative_to(root).as_posix())
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'Nonregular filesystem entry')
        if stat.S_ISREG(mode):
            observed.add(path.relative_to(root).as_posix())
    require(observed == declared | {MANIFEST},
            'Exact member set differs; missing=' + repr(sorted(declared - observed))
            + '; extra=' + repr(sorted(observed - declared - {MANIFEST})))
    return {'files_verified': len(declared), 'payload_bytes_verified': total,
            'exact_member_set_verified': True}


def verify_component(root, name, pin):
    path = checked_path(root, name)
    require(sha_file(path) == pin, 'Frozen component manifest changed: ' + name)
    data = read_json(path)
    files = data.get('files')
    if isinstance(files, dict):
        entries = [dict(metadata, path=member) for member, metadata in files.items()]
    else:
        require(isinstance(files, list), 'Component file list absent')
        entries = files
    names = set()
    for entry in entries:
        member = validate_entry(entry)
        require(member not in names, 'Duplicate component member')
        names.add(member)
        target = checked_path(path.parent, member)
        require(target.stat().st_size == entry['bytes'] and sha_file(target) == entry['sha256'],
                'Frozen component member differs: ' + name + '/' + member)
    return data, len(entries)


def zero_fields(data, names):
    for name in names:
        require(type(data.get(name)) is int and data[name] == 0, 'Zero-execution scope changed: ' + name)


def false_fields(data, names):
    for name in names:
        require(data.get(name) is False, 'False scope invariant changed: ' + name)


def check_science(data):
    require(data.get('base_commit') == BASE
            and data.get('scientific_decision') == 'NO_GO_CONFIRMATION_CURRENT_RECIPE',
            'Scientific decision/source changed')
    require([data.get(key) for key in ('preserved_additive_jobs', 'preserved_V3_fits',
            'preserved_transfer_trajectories', 'preserved_transfer_rows')] == [796, 30, 51, 20400],
            'Historical experiments not preserved separately')
    require(abs(data.get('pilot_error_increase', 0) - 0.02704311988801411) < 1e-15,
            'Adverse pilot result changed')
    false_fields(data, ['R3_accredited', 'rule_changed', 'ablation_promoted'])
    zero_fields(data, ['new_model_fits', 'new_predictions'])


def check_preparation(root, c_manifest):
    false_fields(c_manifest, ['training_executed', 'human_independent_execution_observed',
                             'R3_accredited', 'raw_NPZ_or_checkpoints_included'])
    require(c_manifest.get('fixed_jobs') == 2 and c_manifest.get('planned_comparison_rows') == 110
            and c_manifest.get('reference_CSV_rows') == 935, 'C subset scope changed')
    request = read_json(root / 'reproduction_C_preparation/C_REQUEST.template.json')
    false_fields(request, ['allow_training', 'signature_claimed', 'human_execution_verified_by_software'])
    require(request.get('new_run_id') is None and request.get('recorded_at_utc') is None
            and request.get('operator', {}).get('name') is None
            and request.get('operator', {}).get('independent_of_design') is False,
            'External C declaration invented or template prefilled')
    preflight = read_json(root / 'science/preflight_receipt.json')
    require(preflight.get('status') == 'BLOCKED_PREPARATION_ONLY', 'Prospective status changed')
    false_fields(preflight, ['confirmation_enabled', 'this_program_can_train'])
    zero_fields(preflight, ['new_model_fits', 'new_model_predictions'])


def check_first_archive(data):
    require(data.get('status') == 'ephemeral_full_snapshot_verified_and_scoped_capsule_built'
            and data.get('source_commit') == BASE, 'First archive stage absent/different')
    require(data.get('capsule', {}).get('sha256') == FIRST_ARCHIVE_PIN
            and data['capsule'].get('bytes') == 58823392
            and data['capsule'].get('members') == 18564
            and data.get('counts', {}).get('approved_ONNX_unique_hashes_included') == 6,
            'Original six-graph capsule scope changed')
    require(data.get('historical_snapshot', {}).get('unique_objects_verified') == 9217
            and data['historical_snapshot'].get('ephemeral_only') is True
            and data['historical_snapshot'].get('full_snapshot_uploaded_as_artifact') is False,
            'Ephemeral snapshot relabelled as deposit')
    false_fields(data, ['R3_accredited', 'human_independent_reproduction_claimed',
                        'integral_repository_or_scientific_master_deposited', 'global_license_assigned'])
    zero_fields(data, ['new_model_fits', 'new_model_predictions', 'Zenodo_writes'])


def check_consolidation(data):
    require(data.get('status') == 'scoped_archive_consolidated_with15_exact_approved_ONNX'
            and data.get('source_commit') == BASE and data.get('version') == '0.3.1-REVIEW',
            'Consolidation stage not verified')
    verification = data.get('verification', {})
    require(verification.get('unique_ONNX_graphs_verified') == 15
            and verification.get('historical_ONNX_graphs') == 14
            and verification.get('all_member_hashes_verified') is True
            and data.get('C_addon', {}).get('included') is True,
            'Final scope/extraction/C addon check absent')
    false_fields(data, ['R3_accredited', 'full_historical_master_deposited',
                        'full_historical_master_hydration_repeated', 'V3_serving_enabled',
                        'human_independent_reproduction_claimed', 'global_license_assigned'])
    zero_fields(data, ['new_model_fits', 'new_model_predictions', 'new_calibrations', 'Zenodo_writes'])


def check_release(data, consolidation):
    require(data.get('status') == 'review_release_published_and_public_asset_hashes_verified'
            and data.get('source_repository_cut') == BASE
            and data.get('repository') == 'jassan-bit/DeepLearning_202630'
            and data.get('tag') == 'e3-source-evidence-8d03a73-v0.3.1-review'
            and data.get('prerelease_REVIEW') is True,
            'Review release status/source changed')
    false_fields(data, ['R3_accredited', 'global_license_granted'])
    zero_fields(data, ['new_model_fits', 'Zenodo_writes'])
    require(data.get('launcher_commit') == consolidation.get('launcher_commit'), 'Release launcher mismatch')
    assets = data.get('assets', {})
    archive = consolidation['archive']
    require(set(assets) == {archive['file'], 'capsule_manifest.json', 'consolidation_receipt.json',
                            'RELEASE_NOTES.md'}, 'Release asset set differs')
    for name, asset in assets.items():
        require(asset.get('http') == 200 and type(asset.get('bytes')) is int
                and asset['bytes'] > 0 and re.fullmatch(r'[0-9a-f]{64}', asset.get('sha256', '')),
                'Release asset download verification absent: ' + name)
    actual = assets[archive['file']]
    require(actual['bytes'] == archive['bytes'] and actual['sha256'] == archive['sha256'],
            'Published archive hash/size differs from consolidation')


def verify_packet(root):
    root = Path(root).absolute()
    manifest = read_json(checked_path(root, MANIFEST))
    require(manifest.get('schema_version') == 1
            and manifest.get('kind') == 'e3_resolution_packet_manifest'
            and manifest.get('base_commit') == BASE, 'Unexpected packet schema/source')
    result = verify_inventory(root, manifest.get('files'))
    science, science_count = verify_component(root, 'science/manifest.json', SCIENCE_PIN)
    c_manifest, c_count = verify_component(root, 'reproduction_C_preparation/MANIFEST.json', C_PIN)
    check_science(read_json(root / 'science/scientific_decision.json'))
    check_preparation(root, c_manifest)
    check_first_archive(read_json(root / 'observations/archive_execution_receipt.json'))
    consolidation_path = root / 'observations/consolidation_receipt.json'
    release_path = root / 'observations/review_release_receipt.json'
    final = consolidation_path.is_file()
    release = release_path.is_file()
    if final:
        consolidation = read_json(consolidation_path)
        check_consolidation(consolidation)
    if release:
        require(final, 'Release receipt without executed consolidation')
        check_release(read_json(release_path), consolidation)
    result.update(status='packet_identity_and_declared_scope_verified',
                  base_commit=BASE, science_payloads_verified=science_count,
                  C_payloads_verified=c_count, final_consolidation_receipt_verified=final,
                  public_review_release_receipt_verified=release,
                  scientific_decision='NO_GO_CONFIRMATION_CURRENT_RECIPE', R3_accredited=False,
                  human_reproduction_inferred=False, rights_clearance_inferred=False,
                  model_or_scientific_programs_executed=0, network_requests=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        result = verify_packet(args.root)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({'status': 'rejected', 'error': str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
