"""Build/check a selected, immutable E3 source/audit deposit DRAFT.

No new license, DOI, publication, market inputs or model execution is created.
The source snapshot is pinned separately from the current mutable checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = 'e3-source-audit-1bae73b-v0.1.0-DRAFT'
ARCHIVE = HERE / (PREFIX + '.zip')
MANIFEST = HERE / 'package_manifest.json'
PREPARATION = ['README.md','LICENSE_STATUS.md','ARCHIVE_README.md','ZENODO_FORM.md',
               'source_spec.json','zenodo_metadata.DRAFT.json','rights_matrix.csv',
               'reference_locators.json','restore_references.py','build_package.py']


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def regular(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('regular_non_symlink_file_required: ' + str(path))
    return path


def safe(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name


def materialize():
    spec = json.loads((HERE / 'source_spec.json').read_text())
    metadata = json.loads((HERE / 'zenodo_metadata.DRAFT.json').read_text())
    if spec['source_commit'] != '1bae73b5373c11466b226fb3efab706237b5e68b':
        raise ValueError('new_snapshot_requires_new_version')
    if spec['published'] is not False or spec['doi'] is not None:
        raise ValueError('draft_scope_changed_without_deposit_evidence')
    records = []
    content = {}
    for row in spec['files']:
        name = row['path']
        if not safe(name):
            raise ValueError('unsafe_source_path')
        raw = regular(ROOT / name).read_bytes()
        if len(raw) != row['bytes'] or digest(raw) != row['sha256'] or git_blob(raw) != row['git_blob_sha1']:
            raise ValueError('pinned_source_differs: ' + name)
        content[name] = raw
        records.append({'path':name,'bytes':len(raw),'sha256':digest(raw),
                        'git_blob_sha1':git_blob(raw),'origin':'fixed_repository_source',
                        'source_commit':spec['source_commit'],
                        'license_status':'author_choice_pending'})
    for name in PREPARATION:
        source = regular(HERE / name)
        relative = source.relative_to(ROOT).as_posix()
        raw = source.read_bytes()
        content[relative] = raw
        records.append({'path':relative,'bytes':len(raw),'sha256':digest(raw),
                        'origin':'archival_preparation_not_claimed_in_baseline_commit',
                        'source_commit':None,'license_status':'author_choice_pending'})
    aliases = {'ARCHIVE_README.md':'ARCHIVE_README.md','LICENSE_STATUS.md':'LICENSE_STATUS.md',
               'archive_tools/restore_references.py':'restore_references.py',
               'archive_tools/reference_locators.json':'reference_locators.json'}
    for target, source_name in aliases.items():
        raw = (HERE / source_name).read_bytes()
        content[target] = raw
        records.append({'path':target,'bytes':len(raw),'sha256':digest(raw),
                        'origin':'archival_preparation_convenience_copy',
                        'source_commit':None,'license_status':'author_choice_pending'})
    records.sort(key=lambda r:r['path'])
    if len(content) != len(records):
        raise ValueError('duplicate_archive_path')
    source_manifest = {'schema_version':1,'package':PREFIX,'version':spec['version'],
        'source_commit':spec['source_commit'],'source_tree':spec['source_tree'],
        'scope':spec['scope'],'creators':[spec['creator']],
        'AI_assistance_disclosed':True,'license_status':'author_choice_pending',
        'DOI':None,'DOI_reserved':False,'persistent_deposit_completed':False,
        'human_independent_reproduction_completed':False,
        'new_model_fits':0,'new_model_forwards':0,'files':records}
    content['SOURCE_MANIFEST.json'] = canonical(source_manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name, raw in sorted(content.items()):
            info = zipfile.ZipInfo(PREFIX + '/' + name, date_time=(2026,11,18,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    raw = stream.getvalue()
    manifest = {**source_manifest,'archive':{'name':ARCHIVE.name,'bytes':len(raw),'sha256':digest(raw)},
                'internal_manifest_sha256':digest(content['SOURCE_MANIFEST.json']),
                'archive_members':len(content),
                'numeric_reference_tables_included':False,'market_data_included':False,
                'trained_weights_included':False,'metadata_state':metadata['state']}
    return raw, canonical(manifest), manifest


def verify_existing():
    manifest = json.loads(regular(MANIFEST).read_text())
    raw = regular(ARCHIVE).read_bytes()
    if len(raw) != manifest['archive']['bytes'] or digest(raw) != manifest['archive']['sha256']:
        raise ValueError('archive_integrity_failed')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        expected = {PREFIX + '/' + row['path'] for row in manifest['files']} | {PREFIX + '/SOURCE_MANIFEST.json'}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('archive_members_differ')
        for row in manifest['files']:
            if not safe(row['path']):
                raise ValueError('unsafe_archive_path')
            body = archive.read(PREFIX + '/' + row['path'])
            if len(body) != row['bytes'] or digest(body) != row['sha256']:
                raise ValueError('archive_member_integrity_failed: ' + row['path'])
            if row['origin'] == 'fixed_repository_source' and git_blob(body) != row['git_blob_sha1']:
                raise ValueError('fixed_source_blob_differs')
        internal = archive.read(PREFIX + '/SOURCE_MANIFEST.json')
        if digest(internal) != manifest['internal_manifest_sha256'] or json.loads(internal)['files'] != manifest['files']:
            raise ValueError('internal_manifest_differs')
    if manifest['DOI'] is not None or manifest['persistent_deposit_completed'] is not False:
        raise ValueError('draft_claims_a_deposit')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        manifest = verify_existing()
        status = 'verified_existing_draft_package'
    else:
        raw, manifest_raw, manifest = materialize()
        for path, body in [(ARCHIVE,raw),(MANIFEST,manifest_raw)]:
            if path.exists() and path.read_bytes() != body:
                raise ValueError('preserve_existing_version_create_new_package: ' + str(path))
            if not path.exists():
                with path.open('xb') as stream:
                    stream.write(body)
        verify_existing()
        status = 'created_or_identical_draft_package'
    print(json.dumps({'status':status,**manifest['archive'],'members':manifest['archive_members'],
                      'pinned_source_files':sum(r['origin']=='fixed_repository_source' for r in manifest['files']),
                      'license_status':manifest['license_status'],'DOI':manifest['DOI'],
                      'persistent_deposit_completed':manifest['persistent_deposit_completed']},ensure_ascii=False))


if __name__ == '__main__':
    main()
