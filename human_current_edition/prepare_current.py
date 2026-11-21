"""Prepare a new, isolated final-edition case; never runs it."""
import argparse,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
from run_current import SOURCE_COMMIT,HERE,BASE_MANIFEST,ADDON_MANIFEST,digest,require

def copy_manifest(source,dest):
    m=json.loads((source/'MANIFEST.json').read_text())
    for row in m['files']+[{'path':'MANIFEST.json'}]:
        p=dest/row['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/row['path'],p)
    return m
def prepare(source,case,local_git=None):
    case=Path(case).absolute();require(not case.exists(),'A new case directory is required')
    require(not any(p.is_symlink() for p in (case,*case.parents)),'Symlink destination rejected')
    base=HERE/'frozen_base';addon=HERE/'frozen_addon'
    require(digest(base/'MANIFEST.json')==BASE_MANIFEST and digest(addon/'MANIFEST.json')==ADDON_MANIFEST,'Frozen dependencies differ')
    sys.path.insert(0,str(base));import prepare_external as frozen
    frozen.inspect_kit(base);frozen.inspect_kit(addon);frozen.inspect_kit(HERE);frozen.SOURCE_COMMIT=SOURCE_COMMIT
    case.mkdir();(case/'output').mkdir()
    if local_git:
        local_git=Path(local_git).resolve();work=case/'source';work.mkdir()
        subprocess.run(['git','init',str(work)],check=True,capture_output=True)
        (work/'.git/objects/info/alternates').write_text(str(local_git/'objects')+'\n')
        if (local_git/'shallow').exists():shutil.copyfile(local_git/'shallow',work/'.git/shallow')
        subprocess.run(['git','-C',str(work),'read-tree',SOURCE_COMMIT],check=True,capture_output=True)
        subprocess.run(['git','-C',str(work),'checkout-index','-a'],check=True,capture_output=True)
        subprocess.run(['git','-C',str(work),'update-ref','--no-deref','HEAD',SOURCE_COMMIT],check=True,capture_output=True)
        rows=frozen.source_inventory(work)
    else:
        require(source is not None,'--source-root or --local-git-dir required');source=Path(source).resolve()
        require(not case.is_relative_to(source),'Separate source and case directories required');rows=frozen.source_inventory(source)
        for row in rows:
            p=case/'source'/row['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/row['path'],p)
            require(digest(p)==row['sha256'],'Source copy differs')
    copy_manifest(base,case/'kit');copy_manifest(addon,case/'addon');copy_manifest(HERE,case/'current')
    inventory={'source_commit':SOURCE_COMMIT,'files':rows,'bytes':sum(r['bytes'] for r in rows)}
    (case/'SOURCE_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (case/'HUMAN_REQUEST.json').write_bytes((HERE/'HUMAN_REQUEST.template.json').read_bytes())
    receipt={'status':'fresh_final_edition_case_prepared_not_executed','source_commit':SOURCE_COMMIT,'tracked_files':len(rows),'copied_source_bytes':inventory['bytes'],'source_inventory_sha256':digest(case/'SOURCE_INVENTORY.json'),'human_execution_claimed':False,'new_model_fits':0,'new_model_predictions':0,'new_statistical_resamples':0,'base_manifest_sha256':BASE_MANIFEST,'addon_manifest_sha256':ADDON_MANIFEST,'current_manifest_sha256':digest(HERE/'MANIFEST.json')}
    (case/'PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path);p.add_argument('--local-git-dir',type=Path);p.add_argument('--case-root',type=Path,required=True);a=p.parse_args()
    try:print(json.dumps(prepare(a.source_root,a.case_root,a.local_git_dir),sort_keys=True))
    except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);raise SystemExit(2)
