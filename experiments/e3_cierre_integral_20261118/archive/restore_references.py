"""Acquire separately pinned CSV references; no model runs or redistribution grant."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent
    manifest = json.loads((here / 'reference_locators.json').read_text())
    for row in manifest['files']:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('unsafe_reference_path')
        if args.dry_run:
            print(row['path'], row['bytes'], row['sha256'])
            continue
        target = root / row['path']
        if target.is_symlink():
            raise ValueError('symlink_reference_refused')
        if target.exists():
            raw = target.read_bytes()
        else:
            request = Request(row['url'], headers={'User-Agent': 'E3-Source-Archive-References/1'})
            with urlopen(request, timeout=60) as response:
                raw = response.read(row['bytes'] + 1)
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('reference_bytes_differ: ' + row['path'])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
        print('verified', row['path'])
    print('No market models fitted, no DOI created, no redistribution license granted.')


if __name__ == '__main__':
    main()
