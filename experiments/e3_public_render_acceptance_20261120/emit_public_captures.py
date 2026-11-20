"""Emit only fixed-client public captures, not arbitrary artifact contents.

The original image bytes are preserved. AI assistance declared. No credentials,
model weights, requests or training are involved.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

CHUNK_CHARACTERS = 49152


def capture_paths(root):
    root = root.resolve()
    paths = []
    for directory in ('captures', 'captures_five'):
        folder = root / directory
        if folder.is_symlink():raise ValueError('Capture directory symlink rejected')
        if not folder.exists():continue
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():raise ValueError('Capture symlink rejected')
            if not path.is_file() or path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):continue
            if not path.resolve().is_relative_to(root):raise ValueError('Capture outside declared output')
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.output.resolve(); index = []
    for path in capture_paths(root):
        raw = path.read_bytes()
        if len(raw) > 8 * 1024 * 1024:raise ValueError('Public capture exceeds declared eight MiB limit')
        encoded = base64.b64encode(raw).decode('ascii')
        count = (len(encoded) + CHUNK_CHARACTERS - 1) // CHUNK_CHARACTERS
        entry = {'path': str(path.relative_to(root)), 'bytes': len(raw),
                 'sha256': hashlib.sha256(raw).hexdigest(), 'chunks': count, 'encoding': 'base64',
                 'scope': 'public_Render_capture_generated_by_fixed_acceptance_client'}
        for i in range(count):
            part = encoded[i * CHUNK_CHARACTERS:(i + 1) * CHUNK_CHARACTERS]
            print('E3_PUBLIC_CAPTURE_CHUNK=' + json.dumps({**entry, 'index': i, 'data': part}, sort_keys=True))
        index.append(entry)
    print('E3_PUBLIC_CAPTURE_INDEX=' + json.dumps({'files': index, 'chunk_characters_limit': CHUNK_CHARACTERS}, sort_keys=True))


if __name__ == '__main__':main()
