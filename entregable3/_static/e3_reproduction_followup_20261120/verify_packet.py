"""Read-only payload verification; no data download, fitting or inference."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    doc=json.loads((root/'MANIFEST.sha256.json').read_text())
    for item in doc['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe path')
        payload=(root/rel).read_bytes()
        if len(payload)!=item['bytes'] or hashlib.sha256(payload).hexdigest()!=item['sha256']:
            raise ValueError('Packet payload differs: '+str(rel))
    print(json.dumps({'status':'passed','verified_payloads':len(doc['files']),'new_model_fits':0,'inferences':0,'R3_accredited':False}))


if __name__=='__main__':main()
