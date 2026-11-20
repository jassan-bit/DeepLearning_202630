"""Build a reviewable draft; never upload or publish."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
PAYLOAD=HERE/'payload'
NAME='e3-entregable3-documentary-60a9d988-v0.2.0-DRAFT.zip'

def build(output):
    files={str(p.relative_to(PAYLOAD)):p.read_bytes() for p in sorted(PAYLOAD.rglob('*'))
           if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    manifest={'schema_version':1,'source_commit':'60a9d988af9608b1037c10754e68cac3287277a9',
              'scope':'curated_documentary_supplement_not_integral_repository',
              'rights_and_license_confirmed':False,'publication_authorized':False,
              'AI_assistance_declared':True,
              'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()}}
    content=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    (PAYLOAD/'MANIFEST.json').write_bytes(content);files['MANIFEST.json']=content
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,raw in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,11,20,0,0,0));info.create_system=3
            info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_STORED
            archive.writestr(info,raw)
    raw=output.read_bytes()
    return {'file':output.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'members':len(files),'license_confirmed':False,'Zenodo_published':False}

if __name__=='__main__':
    out=HERE/NAME;result=build(out)
    second=HERE/'rebuild-check.zip';again=build(second)
    assert out.read_bytes()==second.read_bytes();second.unlink()
    spec=importlib.util.spec_from_file_location('verify_current_archive',PAYLOAD/'verify_archive.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    result['integrity']=verifier.verify_zip(out.read_bytes());result['two_builds_byte_identical']=True
    (HERE/'DRAFT_ARCHIVE_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
