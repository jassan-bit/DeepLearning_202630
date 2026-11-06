#!/usr/bin/env python3
"""Acquire only the immutable author files declared by the completion design.

No market data, weights, papers, imports of author code, or fits. TLS and the
configured proxy are retained. An existing unequal file is never overwritten.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'experiments/e2_completion_20261106/design.json'
MAX_BYTES = 4 * 1024 * 1024


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid source path')
    p = PurePosixPath(value)
    if p.is_absolute() or p.as_posix() != value or any(x in {'.', '..'} or ':' in x for x in p.parts):
        raise ValueError('Source path must be a normalized relative path')
    return p


def destination(root, name):
    p = root
    for part in relative(name).parts:
        p = p / part
        if p.is_symlink():
            raise ValueError('Source destinations cannot contain symlinks')
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('Source destination escaped the selected directory')
    return p


def validate(design):
    locations = design['source_locations']
    rows = design['author_sources']
    if not isinstance(locations, dict) or not isinstance(rows, list) or not rows:
        raise ValueError('The design must bind all author sources')
    seen = set()
    for model, source in locations.items():
        relative(model)
        if len(PurePosixPath(model).parts) != 1:
            raise ValueError('Each source tree needs a distinct safe name')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', source['repository']):
            raise ValueError('Invalid repository')
        if not re.fullmatch(r'[0-9a-f]{40}', source['commit']):
            raise ValueError('A full immutable commit is required')
    for row in rows:
        if row['model'] not in locations:
            raise ValueError('Unknown author tree')
        key = (row['model'], str(relative(row['path'])))
        relative(row['repository_path'])
        if key in seen:
            raise ValueError('Duplicate source destination')
        seen.add(key)
        if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('Every source file needs a SHA256')
        if not isinstance(row['bytes'], int) or not 0 <= row['bytes'] <= MAX_BYTES:
            raise ValueError('Source file size outside the permitted bounds')
    if {r['model'] for r in rows} != set(locations):
        raise ValueError('Every declared source tree must contain pinned files')


def fetch(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'raw.githubusercontent.com' or parsed.query or parsed.fragment:
        raise ValueError('Only direct pinned HTTPS source URLs are supported')
    with urlopen(Request(url, headers={'User-Agent': 'E2CompletionSourceAcquisition/1.0'}), timeout=45) as r:
        body = r.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError('Source download exceeds its size limit')
    return body


def acquire(design, output, *, fetcher=fetch):
    validate(design)
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError('The output root must not be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    verified = []
    for row in design['author_sources']:
        source = design['source_locations'][row['model']]
        path = destination(output, row['model'] + '/' + row['path'])
        url = 'https://raw.githubusercontent.com/{}/{}/{}'.format(source['repository'], source['commit'], row['repository_path'])
        if path.exists():
            if not path.is_file() or path.stat().st_size != row['bytes']:
                raise ValueError('Existing source has a different size: ' + row['model'] + '/' + row['path'])
            body = path.read_bytes()
        else:
            body = fetcher(url)
        if len(body) != row['bytes'] or sha256(body).hexdigest() != row['sha256']:
            raise ValueError('Author source hash differs: ' + row['model'] + '/' + row['path'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(body)
        verified.append({'model': row['model'], 'path': row['path'], 'sha256': row['sha256']})
    roots = {model: str((output / model).resolve()) for model in design['source_locations']}
    (output / 'source_roots.json').write_text(json.dumps(roots, indent=2) + '\n')
    receipt = {'schema_version': 1, 'verified_sources': verified,
               'market_payloads_opened': 0, 'weights_downloaded': 0,
               'design_source_identity_sha256': sha256(json.dumps({k: design[k] for k in ('source_locations', 'author_sources')}, sort_keys=True).encode()).hexdigest()}
    (output / 'acquisition_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=DESIGN)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(ROOT):
        parser.error('Author trees must remain outside the project checkout')
    receipt = acquire(json.loads(args.design.read_text()), args.output_dir)
    print(json.dumps({'files_verified': len(receipt['verified_sources']), 'source_roots_json': str(args.output_dir / 'source_roots.json'), 'market_fits': 0}))


if __name__ == '__main__':
    main()
