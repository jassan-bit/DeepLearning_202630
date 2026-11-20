"""Verify this follow-up and its book mirrors without executing experiments."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    home=root/'experiments/e3_followup_20261120'
    manifest=json.loads((home/'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        parts=Path(relative).parts
        if Path(relative).is_absolute() or '..' in parts:raise ValueError('Unsafe path')
        p=root/relative
        if p.is_symlink():raise ValueError('Symlink not permitted:'+relative)
        raw=p.read_bytes()
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise ValueError('Follow-up hash mismatch:'+relative)
    mirrors=0
    for name in manifest['mirrored_experiments']:
        for p in (root/'experiments'/name).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
            twin=root/'book_e3/_static'/name/p.relative_to(root/'experiments'/name)
            if twin.read_bytes()!=p.read_bytes():raise ValueError('Mirror mismatch:'+str(twin))
            mirrors+=1
    decision=json.loads((root/'experiments/e3_science_followup_20261120/decision.json').read_text())
    if decision['R3_accredited'] or decision['scientific_decision']!='NO_GO_CONFIRMATION_CURRENT_RECIPE':
        raise ValueError('Scientific decision changed')
    print(json.dumps({'status':'followup_hashes_and_mirrors_verified',
                      'payloads':len(manifest['files']),'mirrors':mirrors,
                      'new_training_runs_by_verifier':0,'R3_accredited':False}))

if __name__=='__main__':main()
