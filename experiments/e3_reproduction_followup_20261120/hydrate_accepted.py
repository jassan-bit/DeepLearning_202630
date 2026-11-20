"""Download accepted public inputs, never export or fit. AI assistance declared."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root, manifest):
    records=[]
    for item in manifest['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe input path')
        path=Path(root)/rel
        data=path.read_bytes()
        if len(data)!=item['bytes'] or sha(data)!=item['sha256']:
            raise ValueError('Input bytes/hash differ; preserve and investigate: '+item['path'])
        records.append({'path':item['path'],'bytes':len(data),'sha256':sha(data)})
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent
    manifest=json.loads((root/'accepted_input_manifest.json').read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);receipts=[]
    for item in manifest['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe input path')
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            data=dest.read_bytes();mode='existing_hash_verified'
        else:
            req=urllib.request.Request(item['url'],headers={'User-Agent':'E3-accepted-binary-replay/1.0'})
            with urllib.request.urlopen(req,timeout=60) as response:
                data=response.read(item['bytes']+1)
                if response.status!=200:raise ValueError('Expected HTTP200')
            mode='downloaded_public_HTTP200'
        if len(data)!=item['bytes'] or sha(data)!=item['sha256']:
            raise ValueError('Input bytes/hash differ; existing files are not overwritten: '+item['path'])
        if not dest.exists():
            with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as f:
                f.write(data);temporary=Path(f.name)
            os.replace(temporary,dest);dest.chmod(0o644)
        receipts.append({'path':item['path'],'mode':mode,'bytes':len(data),'sha256':sha(data)})
    verify(out,manifest)
    receipt={'status':'passed','observed_at_UTC':datetime.now(timezone.utc).isoformat(),'AI_assistance_declared':True,'inputs':receipts,
             'reexport_executed':False,'new_model_fits':0,'new_calibrations':0}
    (out/'hydration_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
