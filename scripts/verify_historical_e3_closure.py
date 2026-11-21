#!/usr/bin/env python3
"""Verify the frozen historical E3 closure and a strictly append-only design log.

AI assistance declared. The old closure's zero-work counters describe only that
historical component. They are not counters for subsequent development work.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import types

CLOSURE = PurePosixPath('experiments/e3_evidence_close_20261120')
STATIC = PurePosixPath('book_e3/_static/e3_evidence_close_20261120')
MANIFEST_SHA256 = 'fc7baad48d29ee832473275c6283400669aee40d0820b5ebbcb89e121a288d82'
VERIFIER_SHA256 = 'a8ad5a1d06e1aef43faa809301ae9acf55b7af1ad392e48965b8d71ec945e1ac'
APPEND_ONLY = 'experiments/design_log.md'
# edited_root_files in the frozen manifest has no bytes field. This length is
# pinned from its exact original file (SHA below), without changing the manifest.
HISTORICAL_LOG_BYTES = 6025
HISTORICAL_LOG_SHA256 = '99c0d894cacadc93104a112a98c73ac2f68fc6c74a33dfde1517b22f446cc972'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def member(root, relative):
    require(isinstance(relative,str) and bool(relative), 'Invalid manifest member')
    part=PurePosixPath(relative)
    require(not part.is_absolute() and '..' not in part.parts and
            '\\' not in relative and part.as_posix()==relative, 'Unsafe manifest member: '+relative)
    current=root
    for segment in part.parts:
        current=current/segment
        require(not current.is_symlink(), 'Symlinked source/member: '+relative)
    require(current.is_file(), 'Missing source/member: '+relative)
    require(current.resolve().is_relative_to(root), 'Escaped repository: '+relative)
    return current

def link_readonly_source(source, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        require(sha(destination)==sha(source), 'Conflicting overlay destination')
        return
    try:
        os.link(source,destination)
    except OSError:
        shutil.copyfile(source,destination)

def verify_historical(root):
    supplied=Path(root)
    require(not supplied.is_symlink(), 'Repository root is symlinked')
    root=supplied.resolve(strict=True)
    manifest_path=member(root,str(CLOSURE/'closure_manifest.json'))
    require(sha(manifest_path)==MANIFEST_SHA256, 'Frozen closure manifest changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    old_verifier=member(root,str(CLOSURE/'verify_closure.py'))
    require(sha(old_verifier)==VERIFIER_SHA256, 'Frozen closure verifier changed')
    mirror_manifest=member(root,str(STATIC/'closure_manifest.json'))
    require(sha(mirror_manifest)==MANIFEST_SHA256, 'Frozen published manifest changed')
    records=manifest['edited_root_files']
    paths=[r['path'] for r in records]
    require(len(set(paths))==len(paths) and paths.count(APPEND_ONLY)==1, 'Edited-source inventory changed')
    log_record=next(r for r in records if r['path']==APPEND_ONLY)
    require(log_record['sha256']==HISTORICAL_LOG_SHA256, 'Historical log pin changed')
    log=member(root,APPEND_ONLY)
    current_bytes=log.read_bytes()
    require(len(current_bytes)>=HISTORICAL_LOG_BYTES, 'Historical design log truncated')
    prefix=current_bytes[:HISTORICAL_LOG_BYTES]
    require(hashlib.sha256(prefix).hexdigest()==HISTORICAL_LOG_SHA256,
            'Historical design log prefix changed; only tail append is permitted')
    # Check every protected file at its actual current path before using fallback.
    # No absent-protected or source-only mode is exposed by this driver.
    for record in manifest['protected_existing_files']:
        path=member(root,record['path'])
        require(sha(path)==record['sha256'], 'Existing evidence/model artifact changed: '+record['path'])
    with tempfile.TemporaryDirectory(prefix='e3-historical-closure-view-') as temp:
        overlay=Path(temp)
        link_readonly_source(manifest_path,overlay/str(CLOSURE/'closure_manifest.json'))
        link_readonly_source(mirror_manifest,overlay/str(STATIC/'closure_manifest.json'))
        for group in ['closure_payloads','optional_downloads']:
            for record in manifest.get(group,[]):
                relative=record['path']
                for folder in [CLOSURE,STATIC]:
                    name=str(folder/relative)
                    source=member(root,name)
                    # Both copies remain actual immutable bytes, not reconstructed payloads.
                    link_readonly_source(source,overlay/name)
        for record in records:
            source=member(root,record['path'])
            target=overlay/record['path']
            target.parent.mkdir(parents=True,exist_ok=True)
            if record['path']==APPEND_ONLY:
                target.write_bytes(prefix)
            else:
                require(sha(source)==record['sha256'], 'Edited source differs: '+record['path'])
                link_readonly_source(source,target)
        # Execute only the exact pinned verifier source; avoid import bytecode writes.
        module=types.ModuleType('e3_frozen_historical_closure_verifier')
        module.__file__=str(old_verifier)
        exec(compile(old_verifier.read_bytes(),str(old_verifier),'exec'),module.__dict__)
        historical=module.verify(overlay,base_root=root,source_only=False,allow_missing_protected=False)
    require(historical['protected_files_absent']==0 and historical['protected_manifest_files_fully_verified'],
            'Historical protected evidence is incomplete')
    tail=current_bytes[HISTORICAL_LOG_BYTES:]
    return {
        'status':'verified_historical_E3_closure_with_append_only_design_log',
        'frozen_manifest_sha256':MANIFEST_SHA256,
        'frozen_verifier_sha256':VERIFIER_SHA256,
        'historical_component_scope_only':historical,
        'scope_note':'Historical zero-work counters apply only to the original frozen closure; later V4 development has20newfits/1210predictions in its separate packet.',
        'current_design_log':{'path':APPEND_ONLY,'bytes':len(current_bytes),'sha256':hashlib.sha256(current_bytes).hexdigest(),
                              'historical_prefix_bytes':HISTORICAL_LOG_BYTES,'historical_prefix_sha256':HISTORICAL_LOG_SHA256,
                              'appended_bytes':len(tail),'appended_sha256':hashlib.sha256(tail).hexdigest()},
        'source_payloads_and_mirrors_reconstructed':False,
        'missing_protected_allowed':False,
        'new_experiments_executed_by_this_verifier':False,
        'AI_assistance_declared':True,
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(json.dumps(verify_historical(args.repo_root),indent=2,sort_keys=True))

if __name__=='__main__':
    main()
