"""Prepare an isolated human A/B case and commands; never execute models."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE='python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f'
q=lambda value:shlex.quote(str(value))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--case-root',required=True,type=Path)
    parser.add_argument('--level',choices=['A','B'],required=True)
    args=parser.parse_args();source=args.source_root.resolve();case=args.case_root.resolve()
    if case.exists():raise ValueError('New isolated case directory required; preserve all earlier results')
    if args.level=='A':
        required=['deployment/science/Dockerfile','deployment/science/requirements.lock.txt',
                  'experiments/e3_remediacion_20261118/reproduction/reproduce.py']
        absent=[name for name in required if not (source/name).is_file()]
        if absent:raise ValueError('Complete pinned A checkout required; missing: '+', '.join(absent))
    case.mkdir(parents=True)
    subprocess.run([sys.executable,str(HERE/'inspect_kit.py'),'--source-root',str(source),'--output',str(case/'source_inspection.json')],check=True)
    (case/'output').mkdir();commands=['#!/usr/bin/env bash','set -euo pipefail']
    comparisons=[]
    if args.level=='B':
        original=source/'experiments/e3_reproduction_followup_20261120';payload=case/'payload';payload.mkdir()
        for name in ['hydrate_accepted.py','replay_accepted_v3.py','accepted_input_manifest.json','requirements-runtime.lock','Dockerfile']:
            shutil.copyfile(original/name,payload/name);(payload/name).chmod(0o644)
        commands += [
            f'python3 {q(payload/"hydrate_accepted.py")} --output {q(case/"inputs")}',
            f'python3 {q(HERE/"inspect_kit.py")} --source-root {q(source)} --inputs {q(case/"inputs")} --output {q(case/"hydrated_input_inspection.json")}',
            f'python3 -m pip download --only-binary=:all: --no-deps --require-hashes -r {q(payload/"requirements-runtime.lock")} --dest {q(payload/"wheelhouse")}',
            f'docker pull {q(BASE)}',
            f'docker build --network none --pull=false --no-cache -t e3-human-accepted-v3:local {q(payload)}',
            'docker run --rm --network none --read-only --cap-drop=ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=32m --user "$(id -u):$(id -g)" --cpus 1 --memory 256m --pids-limit 64 '+
            f'-v {q(str(payload)+":/payload:ro")} -v {q(str(case/"inputs")+":/inputs:ro")} -v {q(str(case/"output")+":/output:rw")} e3-human-accepted-v3:local --inputs /inputs --output /output'
        ]
        comparisons=[('B.accepted55',original/'execution/discrepancies.csv',case/'output/discrepancies.csv')]
    else:
        commands += [
            f'cd {q(source)}',
            f'python3 experiments/e3_remediacion_20261118/reproduction/reproduce.py fetch --data {q(case/"inputs")}',
            f'docker build -f deployment/science/Dockerfile -t e3-human-scientific-replay:v1 .',
            'docker run --rm --read-only --network none --cap-drop=ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,size=64m --memory=1g --cpus=1 --pids-limit=128 --user "$(id -u):$(id -g)" '+
            f'--mount {q("type=bind,src="+str(case/"inputs")+",dst=/inputs,readonly")} --mount {q("type=bind,src="+str(case/"output")+",dst=/output")} e3-human-scientific-replay:v1'
        ]
        manifest=json.loads((source/'book_e3/_static/e3_remediacion_20261118/reproduction/reference_manifest.json').read_bytes())
        comparisons=[('A.'+table['name'],source/table['reference'],case/'output'/table['observed']) for table in manifest['tables']]
    for name,reference,observed in comparisons:
        commands.append(f'python3 {q(HERE/"compare_csv.py")} --profile {q(name)} --reference {q(reference)} --observed {q(observed)} --output {q(case/("comparison_"+name+".json"))}')
    (case/'commands.sh').write_text('\n'.join(commands)+'\n');(case/'commands.sh').chmod(0o755)
    shutil.copyfile(HERE/'INDEPENDENT_REPRODUCTION_FORM.md',case/'INDEPENDENT_REPRODUCTION_FORM.md')
    report={'status':'case_and_commands_prepared_not_executed','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
            'level':args.level,'source_root':str(source),'case_root':str(case),'source_commit':'8d03a73d529eb3f9b2e89992b4488cfeb88a99e6',
            'commands_path':str(case/'commands.sh'),'new_model_fits':0,'new_model_inferences':0,
            'human_execution_observed':False,'level_C_training_reproduced':False,'AI_assistance_declared':True}
    (case/'PREPARATION_RECEIPT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
