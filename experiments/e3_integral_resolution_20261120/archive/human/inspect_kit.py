"""Inspect frozen A/B source receipts and optional hydrated B inputs; no models."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();home=Path(__file__).resolve().parent
    if args.output.exists():raise ValueError('Choose a new receipt path')
    pinned=json.loads((home/'SOURCE_RECEIPTS.json').read_bytes())
    report={'status':'failed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_commit':pinned['source_commit'],'source_files_checked':0,'B_input_files_checked':0,
            'new_model_fits':0,'new_model_inferences':0,'statistical_reanalysis_executed':False,
            'human_independent_reproduction_claimed':False,'level_C_training_executed':False,
            'AI_assistance_declared':True,'errors':[]}
    try:
        root=args.source_root.resolve()
        for name,wanted in pinned['source_files'].items():
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe source path')
            source=root/name
            if source.is_symlink() or not source.is_file():raise ValueError('Source absent/symlink:'+name)
            raw=source.read_bytes()
            if len(raw)!=wanted['bytes'] or hashlib.sha256(raw).hexdigest()!=wanted['sha256']:
                raise ValueError('Pinned source mismatch:'+name)
            report['source_files_checked']+=1
        if args.inputs is not None:
            manifest=json.loads((root/'experiments/e3_reproduction_followup_20261120/accepted_input_manifest.json').read_bytes())
            for wanted in manifest['files']:
                path=args.inputs/wanted['path']
                if path.is_symlink() or not path.is_file():raise ValueError('B input absent/symlink:'+wanted['path'])
                raw=path.read_bytes()
                if len(raw)!=wanted['bytes'] or hashlib.sha256(raw).hexdigest()!=wanted['sha256']:
                    raise ValueError('B input mismatch:'+wanted['path'])
                report['B_input_files_checked']+=1
        report['status']='frozen_sources_and_requested_inputs_verified_without_execution'
    except (OSError,ValueError,KeyError,TypeError) as error:
        report['errors'].append({'type':type(error).__name__,'message':str(error)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if not report['errors'] else 2


if __name__=='__main__':raise SystemExit(main())
