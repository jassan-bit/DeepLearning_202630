"""Verify a curated documentary archive; no network, models or training."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

SOURCE='60a9d988af9608b1037c10754e68cac3287277a9'
SERVICE='feb83dbc08197bde7076857bd8ff475de197efd2'

def require(value,message):
    if not value:raise ValueError(message)

def verify_members(members):
    require('MANIFEST.json' in members,'Manifest missing')
    manifest=json.loads(members['MANIFEST.json'])
    require(manifest['source_commit']==SOURCE,'Wrong source cut')
    require(set(members)==set(manifest['files'])|{'MANIFEST.json'},'Unexpected or missing member')
    for name,record in manifest['files'].items():
        path=PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,'Unsafe member')
        raw=members[name]
        require(len(raw)==record['bytes'],'Size mismatch:'+name)
        require(hashlib.sha256(raw).hexdigest()==record['sha256'],'SHA mismatch:'+name)
    prefix='repository/experiments/e3_render_public_acceptance_20261120/'
    summary=json.loads(members[prefix+'public_acceptance_summary.json'])
    require(summary['status']=='public_Render_acceptance_passed','Public acceptance not passed')
    require(summary['deployed_service_commit']==SERVICE,'Wrong deployed source')
    require(summary['five_model_count']==5 and summary['fourteen_counts']['API_predictions']==14,'Wrong model counts')
    require(not summary['R3_accredited'] and not summary['V3_enabled'],'Scientific scope changed')
    fields=json.loads(members['ZENODO_FIELDS.DRAFT.json'])
    require(fields['record_id']==24356217,'Wrong reserved record')
    require(not fields['DOI_publication_claimed'],'Publication must not be inferred from an archive')
    return {'status':'curated_archive_integrity_verified','members':len(members),
            'payloads':len(manifest['files']),'source_commit':SOURCE,
            'rights_and_license_confirmed':manifest['rights_and_license_confirmed'],
            'full_training_archive':False,'R3_accredited':False,'new_model_fits':0}

def verify_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)),'Duplicate ZIP member')
        require(archive.testzip() is None,'CRC failure')
        require(sum(z.file_size for z in archive.infolist())<20_000_000,'Archive exceeds declared scope')
        for entry in archive.infolist():
            require((entry.external_attr>>16)&0o170000 !=0o120000,'Symbolic link forbidden')
        return verify_members({name:archive.read(name) for name in names})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',nargs='?',type=Path)
    p.add_argument('--directory',type=Path)
    a=p.parse_args()
    if a.directory:
        root=a.directory.resolve();members={}
        for f in root.rglob('*'):
            if '__pycache__' in f.parts or f.suffix=='.pyc':continue
            if f.is_file():
                require(not f.is_symlink(),'Symbolic link forbidden')
                members[str(f.relative_to(root))]=f.read_bytes()
        result=verify_members(members)
    else:
        require(a.archive is not None,'Supply a ZIP or --directory')
        raw=a.archive.read_bytes();result=verify_zip(raw)
        result.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
