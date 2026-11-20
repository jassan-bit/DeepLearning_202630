"""Read-only verification of exact candidate bytes and every ZIP member."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def verify(archive_path, deposit_manifest):
    expected = deposit_manifest['files'].get(archive_path.name)
    if expected is None:
        raise ValueError('archive name not declared')
    raw = archive_path.read_bytes()
    if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise ValueError('archive bytes do not match external deposit manifest')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate ZIP members')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('unsafe ZIP member path')
        manifest = json.loads(archive.read('MANIFEST.json'))
        if set(names) != set(manifest['files']) | {'MANIFEST.json'}:
            raise ValueError('ZIP member inventory differs from internal manifest')
        if len(names) != expected['members']:
            raise ValueError('ZIP member count differs from external manifest')
        for name, record in manifest['files'].items():
            body = archive.read(name)
            if len(body) != record['bytes'] or hashlib.sha256(body).hexdigest() != record['sha256']:
                raise ValueError('ZIP member hash mismatch: ' + name)
    return {'status': 'passed_exact_candidate_and_member_hashes',
            'archive_bytes': len(raw), 'archive_sha256': hashlib.sha256(raw).hexdigest(),
            'members': len(names), 'payload_members': len(manifest['files']),
            'scope': manifest['scope'], 'Zenodo_writes': 0,
            'rights_confirmed': False, 'integral_training_reproduced': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--deposit-manifest', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.archive, json.loads(args.deposit_manifest.read_bytes()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
