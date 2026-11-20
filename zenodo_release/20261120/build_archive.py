"""Rebuild the selected-members archive deterministically; never publish."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
NAME='e3-entregable3-documentary-60a9d988-v0.2.0.zip'

def build(root,output):
    raw_manifest=(root/'MANIFEST.json').read_bytes()
    manifest=json.loads(raw_manifest)
    members={'MANIFEST.json':raw_manifest}
    for name,record in manifest['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe member')
        path=root/p
        if path.is_symlink():raise ValueError('Symlink member')
        raw=path.read_bytes()
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Source member differs:'+name)
        members[name]=raw
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,raw in sorted(members.items()):
            info=zipfile.ZipInfo(name,(2026,11,20,0,0,0));info.create_system=3
            info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_STORED
            archive.writestr(info,raw)
    raw=output.read_bytes()
    return {'file':output.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'members':len(members),'rights_and_licenses_confirmed':True,'Zenodo_published_at_packaging':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE.parent/NAME)
    args=parser.parse_args()
    print(json.dumps(build(HERE,args.output)))
