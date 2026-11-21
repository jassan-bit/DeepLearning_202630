"""Prepare a comprehensive case, or attach to an unexecuted base-v1 case.

Never executes the book, any model or training. Old frozen kits stay unchanged.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE_KIT=HERE.parent if (HERE.parent/'prepare_external.py').is_file() else HERE.parent/'kit'
sys.path.insert(0,str(BASE_KIT))
from prepare_external import prepare, inspect_kit, digest, require, request_template, SOURCE_COMMIT
BASE_MANIFEST_SHA256='a35a5f56d6288ad5640f347b59415f00968277dceb96096fda0dc83db75b0206'

def attach(case):
    case=Path(case).resolve();require(case.is_dir(),'Prepared case absent')
    require(digest(BASE_KIT/'MANIFEST.json')==BASE_MANIFEST_SHA256,'Frozen base-kit manifest differs')
    inspect_kit(BASE_KIT);checked=inspect_kit(HERE)
    require(digest(case/'kit/MANIFEST.json')==BASE_MANIFEST_SHA256,'Case does not contain frozen base kit')
    require(not list((case/'output').iterdir()),'Case already has output; create a new case')
    require(not (case/'addon').exists() and not (case/'ADDON_PREPARATION.json').exists(),'Addon/case must be new')
    require(json.loads((case/'PREPARATION.json').read_text())['source_commit']==SOURCE_COMMIT,'Case source edition differs')
    require(json.loads((case/'HUMAN_REQUEST.json').read_text())==request_template(),'A person already edited this request; prepare a new comprehensive case before declaring scope')
    dest=case/'addon';dest.mkdir()
    rows=json.loads((HERE/'MANIFEST.json').read_text())['files']+[{'path':'MANIFEST.json'}]
    for row in rows:
        out=dest/row['path'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/row['path'],out)
    copied=inspect_kit(dest)
    (case/'HUMAN_REQUEST.json').write_bytes((HERE/'HUMAN_REQUEST.template.json').read_bytes())
    receipt={'status':'comprehensive_addon_prepared_not_executed','source_commit':SOURCE_COMMIT,'base_manifest_sha256':BASE_MANIFEST_SHA256,'addon':checked,'copied_addon':copied,'principal_scope':'additive19x4-clean-plus-development-V3','human_execution_claimed':False,'new_model_fits':0,'new_model_predictions':0}
    (case/'ADDON_PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path);p.add_argument('--case-root',type=Path,required=True);p.add_argument('--attach-to-unexecuted-case',action='store_true');a=p.parse_args()
    if not a.attach_to_unexecuted_case:
        require(a.source_root is not None,'New case requires --source-root')
        require(digest(BASE_KIT/'MANIFEST.json')==BASE_MANIFEST_SHA256,'Frozen base manifest differs')
        prepare(a.source_root,a.case_root,BASE_KIT)
    print(json.dumps(attach(a.case_root),sort_keys=True))
if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as e:print('Extended preparation blocked: '+str(e),file=sys.stderr);raise SystemExit(2)
