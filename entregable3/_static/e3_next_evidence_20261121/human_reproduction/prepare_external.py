"""Create an isolated book/principal-table case; default never executes it.

Uses a complete clean checkout of the pinned edition, not a filtered source
capsule pretending to contain all book assets. No training or model imports.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path, PurePosixPath

SOURCE_COMMIT='f5c960eefedb5fc49b313dd795b1ec84e784bd78'
SOURCE_REPO='https://github.com/jassan-bit/DeepLearning_202630.git'
BASE_IMAGE='python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f'
METRICS='book_e3/_static/e3_evidence_close_20261120/pilot/generated/metrics_recomputed.csv'
ARCHIVE={'url':'https://github.com/jassan-bit/DeepLearning_202630/releases/download/e3-source-evidence-8d03a73-v0.3.1-review/e3-source-evidence-8d03a73-v0.3.1-REVIEW.tar.gz','bytes':72823602,'sha256':'2c9b5d0ee865683ee1555ee7f745ade79aaf99446a93c3958e3943935b0d49eb','scope':'reviewed source8d03 + evidence +15 approved unique ONNX graphs; not full experimental master or complete f5c book checkout'}

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE)

def source_inventory(root):
    root=Path(root).resolve();require(root.is_dir(),'Source checkout absent')
    require(git(root,'rev-parse','HEAD').decode().strip()==SOURCE_COMMIT,'Source commit differs from the frozen edition')
    rows=[]
    for item in git(root,'ls-tree','-rlz','--full-tree',SOURCE_COMMIT).split(b'\0'):
        if not item:continue
        meta,name=item.split(b'\t',1);mode,kind,blob,size=meta.decode().split();name=name.decode('utf-8')
        rel=PurePosixPath(name);require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,'Unsafe source path')
        require(kind=='blob' and mode in ('100644','100755'),'Symlink/submodule/nonregular tracked source not admitted')
        p=root/name;require(p.is_file() and not p.is_symlink(),'Tracked regular file absent')
        for ancestor in p.parents:
            if ancestor==root:break
            require(not ancestor.is_symlink(),'Symlink source ancestor')
        n=int(size);require(p.stat().st_size==n,'Tracked source size mismatch: '+name)
        h=hashlib.sha1(b'blob '+str(n).encode()+b'\0');h256=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block);h256.update(block)
        require(h.hexdigest()==blob,'Tracked working bytes differ from Git object: '+name)
        rows.append({'path':name,'bytes':n,'git_blob_sha1':blob,'sha256':h256.hexdigest(),'mode':mode})
    require(rows and any(r['path']=='requirements-book.lock.txt' for r in rows),'Incomplete source tree')
    return rows

def inspect_kit(kit):
    manifest=json.loads((kit/'MANIFEST.json').read_text());require(manifest.get('schema_version')==1,'Kit manifest schema')
    seen=set()
    for r in manifest['files']:
        name=r['path'];p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and name not in seen,'Unsafe/duplicate kit path');seen.add(name)
        path=kit/name;require(not path.is_symlink() and path.is_file(),'Missing regular kit member')
        require(path.stat().st_size==r['bytes'] and digest(path)==r['sha256'],'Kit member differs: '+name)
    return {'members_verified':len(seen),'manifest_sha256':digest(kit/'MANIFEST.json')}

def request_template():
    return {'schema_version':1,'mode':'human-external','source_commit':SOURCE_COMMIT,'principal_table':'V3-development-seven-recipes-section8.19',
        'operator':{'name':None,'contact_or_identifier':None,'relationship_to_code':None,'independent_of_code':None},
        'scope_accepted_by_operator':None,'scope_adequacy_explanation':None,'declared_before_execution_utc':None,
        'tolerances':{field:{'absolute':None,'relative':None} for field in ['mean_individual_RMSE','SD_individual_RMSE','mean_individual_MAE','mean_bias_prediction_minus_target']},
        'resources':{'cpus':2,'memory_bytes':2147483648,'timeout_seconds':1800,'network_during_execution':False},
        'human_attestation_after_execution_required':True,'fresh_training_requested':False}

def prepare(source,case,kit,archive=None):
    source=source.resolve();case=case.absolute();kit=kit.resolve()
    require(not case.exists(),'Case must be new; no resume/overwrite of an old run')
    require(not any(p.is_symlink() for p in (case,*case.parents)),'Symlink case path/ancestor rejected')
    require(not case.is_relative_to(source) and not source.is_relative_to(case),'Case/source must be separate')
    verified=inspect_kit(kit);rows=source_inventory(source);total=sum(r['bytes'] for r in rows)
    require(total<=3_000_000_000,'Source checkout exceeds declared3GB input budget')
    case.parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(case.parent).free>=total+1_000_000_000,'Insufficient free disk: source copy plus1GB output reserve needed')
    archive_verified=False
    if archive:
        require(archive.stat().st_size==ARCHIVE['bytes'] and digest(archive)==ARCHIVE['sha256'],'Review archive differs');archive_verified=True
    case.mkdir();(case/'output').mkdir();(case/'kit').mkdir()
    for row in json.loads((kit/'MANIFEST.json').read_text())['files']+[{'path':'MANIFEST.json'}]:
        p=case/'kit'/row['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(kit/row['path'],p)
    for row in rows:
        out=case/'source'/row['path'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/row['path'],out)
        require(digest(out)==row['sha256'],'Copied source differs');os.chmod(out,0o755 if row['mode']=='100755' else 0o644)
    # Costly saved results and their generators are retained; built HTML and
    # caches are untracked and never copied into the new source tree.
    require(not (case/'source/book_e3/_build').exists(),'Source unexpectedly versions a built book')
    (case/'SOURCE_INVENTORY.json').write_text(json.dumps({'source_commit':SOURCE_COMMIT,'files':rows,'bytes':total},indent=2)+'\n')
    (case/'HUMAN_REQUEST.json').write_text(json.dumps(request_template(),indent=2)+'\n')
    receipt={'status':'fresh_case_prepared_not_executed','source_commit':SOURCE_COMMIT,'tracked_files':len(rows),'copied_source_bytes':total,
        'source_inventory_sha256':digest(case/'SOURCE_INVENTORY.json'),'kit':verified,'review_archive':dict(ARCHIVE,locally_verified=archive_verified),
        'case_directory':str(case),'new_model_fits':0,'new_model_predictions':0,'human_execution_claimed':False,
        'original_source_modified':False,'Docker_build_required':True}
    (case/'PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path);p.add_argument('--case-root',type=Path);p.add_argument('--review-archive',type=Path);p.add_argument('--inspect-kit',action='store_true');a=p.parse_args();kit=Path(__file__).resolve().parent
    if a.inspect_kit:result=inspect_kit(kit)
    else:
        require(a.source_root and a.case_root,'--source-root and --case-root required')
        result=prepare(a.source_root,a.case_root,kit,a.review_archive)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as e:print('Preparation blocked: '+str(e),file=sys.stderr);raise SystemExit(2)
