"""Build the versioned operational data extension without editing engine A.

AI-assisted implementation. Files are copied (not linked) from the hash-verified
historical manifest; candidate, bank, model, trainer and statistics are pinned.
This script acquires no market data and performs no model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_A_COMMIT = 'c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c'
SOURCE_A_MANIFEST_SHA256 = 'cd96def4f0a792d0d26d82f94048b0dbc9bf10b35cc7d3a561cf0e0db6d5b12b'
PREFIX = 'experiments/e3_hypothesis_adequacy_20261122/runtime'
IMMUTABLE = ('configs/CANDIDATE_IDENTITY.json','configs/FINAL_SEED_BANK.json',
             'engine/models.py','engine/training.py','engine/recipes.json',
             'engine/statistics.py','engine/power.py','engine/controller.py',
             'configs/protocol.json','configs/adequacy.json','configs/power_design.json',
             'vendor/SOURCE_MANIFEST.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    assert source.resolve() != destination.resolve()
    original = source/'SOURCE_MANIFEST.json'
    assert sha(original) == SOURCE_A_MANIFEST_SHA256
    manifest = json.loads(original.read_text())
    for relative, expected in manifest['files'].items():
        a, b = source/relative, destination/relative
        assert sha(a) == expected, relative
        b.parent.mkdir(parents=True,exist_ok=True)
        if not b.exists():
            shutil.copy2(a,b)
    (destination/'HISTORICAL_SOURCE_A_MANIFEST.json').write_bytes(original.read_bytes())
    (destination/'docs/HISTORICAL_ENGINE_A_README.md').write_bytes((source/'README.md').read_bytes())
    return manifest


def verify_preservation(source, destination):
    source, destination = Path(source), Path(destination)
    result = {}
    original = json.loads((source/'SOURCE_MANIFEST.json').read_text())
    for relative, expected in original['files'].items():
        assert sha(source/relative) == expected, 'Historical source A changed: '+relative
    for relative in IMMUTABLE:
        assert sha(source/relative) == sha(destination/relative), relative
        result[relative] = sha(destination/relative)
    return dict(source_A_commit=SOURCE_A_COMMIT, source_A_manifest_sha256=SOURCE_A_MANIFEST_SHA256,
                unchanged_algorithm_and_science_files=result,
                source_A_files_verified=len(original['files']),
                new_market_acquisitions=0,new_market_fits=0,test_opened=False)


def manifest(destination):
    destination = Path(destination)
    excluded = {'SOURCE_MANIFEST.json','PUBLIC_SOURCE_ANCHOR.json',
                'E3_Hipotesis_Adecuacion_Crypto_GitHub.ipynb',
                'E3_Hipotesis_Adecuacion_GitHub.ipynb','start_colab.py'}
    paths = sorted(p for p in destination.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
                   and p.relative_to(destination).as_posix() not in excluded)
    assert not any(p.is_symlink() for p in paths)
    obj = dict(schema_version=1,algorithm='sha256',
        scope='Versioned crypto data acquisition/admission extension; no model, seed-bank, trainer or statistical-rule changes; not a retrospective preregistration',
        historical_source_A_commit=SOURCE_A_COMMIT,
        files={p.relative_to(destination).as_posix():sha(p) for p in paths},
        sizes={p.relative_to(destination).as_posix():p.stat().st_size for p in paths})
    raw=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
    (destination/'SOURCE_MANIFEST.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-a',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    prepare(args.source_a,args.runtime)
    proof=verify_preservation(args.source_a,args.runtime)
    (args.runtime/'SOURCE_A_PRESERVATION.json').write_text(json.dumps(proof,indent=2)+'\n')
    if args.finalize:
        proof['manifest_sha256']=manifest(args.runtime)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
