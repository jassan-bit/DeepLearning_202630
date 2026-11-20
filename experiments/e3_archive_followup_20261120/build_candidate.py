"""Build a limited documentary Zenodo candidate locally; never upload/publish."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = 'e3-documentary-followup-20261120-DRAFT.zip'


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def selected_files(root=ROOT):
    names = ['README.md', 'AUTHOR_RIGHTS_AND_AUTHORIZATION.md',
             'rights_by_component.csv', 'CITATION.cff.DRAFT',
             'zenodo_fields.DRAFT.json', 'source_inventory.json',
             'external_artifact_locators.json', 'verify_zenodo_deposit.py',
             'verify_candidate.py', 'build_candidate.py', 'test_archive_integrity.py']
    paths = [root / name for name in names]
    paths.extend(sorted((root / 'evidence').glob('*.json')))
    paths.extend(sorted((root / 'evidence').glob('*.csv')))
    if not paths:
        raise ValueError('empty candidate')
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError('candidate source missing or symlink: ' + str(path))
    return paths


def build(output_dir, root=ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)
    members = {str(path.relative_to(root)): path.read_bytes() for path in selected_files(root)}
    manifest = {
        'schema_version': 1,
        'scope': 'limited_documentary_supplement_not_integral_training_archive',
        'cut_commit': '7db8040619f602d7c52ae0c8bf54f7b6f5b5110e',
        'AI_assistance_declared': True,
        'Zenodo_publication_authorized': False,
        'DOI_publication_claimed': False,
        'excluded': ['OHLCV', 'checkpoints', 'ONNX_binaries', 'mass_predictions',
                     'Python_wheels', 'Docker_images', 'full_paper_bodies'],
        'files': {name: {'bytes': len(raw), 'sha256': sha256(raw)}
                  for name, raw in sorted(members.items())},
        'manifest_excludes_itself': True,
    }
    members['MANIFEST.json'] = encoded(manifest)
    archive_path = output_dir / ARCHIVE_NAME
    # ZIP_STORED avoids zlib version dependence. Fixed timestamp and modes make
    # identical candidate members produce identical archive bytes.
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    raw = archive_path.read_bytes()
    deposit = {
        'schema_version': 1, 'record_id': 24356217,
        'reserved_doi': '10.5281/zenodo.24356217',
        'scope': manifest['scope'], 'publication_authorized': False,
        'license_confirmed': False,
        'files': {ARCHIVE_NAME: {'bytes': len(raw), 'sha256': sha256(raw),
                                'members': len(members)}},
        'deposit_manifest_is_external_and_not_self_hashed': True,
        'verification_scope': 'Candidate ZIP only; rights and integral scientific archive not verified.',
    }
    (output_dir / 'deposit_manifest.json').write_bytes(encoded(deposit))
    print(json.dumps({'candidate': str(archive_path), **deposit}, ensure_ascii=False))
    return archive_path, deposit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.output_dir)
