"""Verify saved DOI/archive evidence and book mirrors without public requests."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

NAME='e3_zenodo_publication_20261120'
SHA='aa59b73311624f2a92b4497f98bfe182e3c2c320c0e0b092fefee5686bce8349'
SOURCE='60a9d988af9608b1037c10754e68cac3287277a9'

def require(ok,message):
    if not ok:raise ValueError(message)

def verify(root):
    home=root/'experiments'/NAME;mirror=root/'book_e3/_static'/NAME
    manifest=json.loads((home/'manifest.json').read_bytes())
    for name,wanted in manifest['files'].items():
        p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'Unsafe path')
        path=home/name;require(not path.is_symlink(),'Symbolic link')
        raw=path.read_bytes()
        require(len(raw)==wanted['bytes'] and hashlib.sha256(raw).hexdigest()==wanted['sha256'],'Payload differs:'+name)
        require((mirror/name).read_bytes()==raw,'Mirror differs:'+name)
    require((home/'manifest.json').read_bytes()==(mirror/'manifest.json').read_bytes(),'Manifest mirror differs')
    report=json.loads((home/'receipts/verification.json').read_bytes())
    require(report['status']=='public_DOI_and_exact_final_archive_verified','Public DOI not verified')
    require(report['public_API']['http']==200 and report['DOI_resolution']['http']==200,'Origin response failed')
    require(report['DOI_resolution']['effective_url']=='https://zenodo.org/records/24356217','Wrong DOI target')
    record_raw=(home/'receipts/public_record.json').read_bytes();record=json.loads(record_raw)
    require(hashlib.sha256(record_raw).hexdigest()==report['public_API']['response_sha256'],'API response differs')
    require(record['status']=='published' and record['doi']=='10.5281/zenodo.24356217','Wrong public state')
    raw=(home/'archive/e3-entregable3-documentary-60a9d988-v0.2.0.zip').read_bytes()
    require(len(raw)==916861 and hashlib.sha256(raw).hexdigest()==SHA,'Deposited ZIP differs')
    require(hashlib.md5(raw).hexdigest()=='61bc587e0e6ea29ab4e3356e2e8b16ef','ZIP MD5 differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(archive.testzip() is None and len(archive.namelist())==66,'ZIP CRC/inventory differs')
        members=json.loads(archive.read('MANIFEST.json'))
        require(set(archive.namelist())==set(members['files'])|{'MANIFEST.json'},'ZIP member set differs')
        for name,wanted in members['files'].items():
            data=archive.read(name)
            require(len(data)==wanted['bytes'] and hashlib.sha256(data).hexdigest()==wanted['sha256'],'ZIP member differs')
        inventory=json.loads(archive.read('SOURCE_INVENTORY.json'))
        require(inventory['source_commit']==SOURCE and len(inventory['source_files'])==53,'Wrong archive source scope')
        require(json.loads(archive.read('OWNER_DECISION.json'))['Zenodo_publication_executed'] is False,
                'Historical packaging decision changed')
    matrix=json.loads((home/'requirements_matrix_current.json').read_bytes())
    prior_raw=(root/matrix['source_matrix_path']).read_bytes();prior=json.loads(prior_raw)
    require(hashlib.sha256(prior_raw).hexdigest()==matrix['source_matrix_sha256'],'Prior matrix differs')
    require(matrix['changed_ids']==[15,16] and len(matrix['rows'])==17,'Wrong matrix change scope')
    for i in list(range(14))+[16]:require(matrix['rows'][i]==prior['rows'][i],'Unrelated matrix row changed')
    require(not report['R3_accredited'] and not matrix['R3_accredited'],'R3 status changed')
    require(report['new_model_fits']==report['new_model_predictions']==0,'New model work claimed')
    require(record['metadata'].get('version') is None and matrix['DOI_metadata_correction_pending'],
            'Pending metadata correction lost')
    return {'status':'published_DOI_book_packet_verified','payloads':len(manifest['files']),
            'ZIP_bytes':916861,'ZIP_SHA256':SHA,'ZIP_member_hashes_verified':65,
            'selected_source_copies':53,'matrix_rows':17,'matrix_changed_ids':[15,16],
            'public_DOI_verified':True,'metadata_corrections_pending':True,
            'R3_accredited':False,'new_model_fits':0,'network_requests':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.root.resolve()),ensure_ascii=False))
