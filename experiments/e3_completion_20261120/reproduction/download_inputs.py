"""Recover only immutable external inputs with byte and SHA256 verification."""
from pathlib import Path
import argparse,hashlib,json,os,tempfile,urllib.request

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--include-additional',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parent;out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'external_input_manifest.json').read_text());done=[]
    for item in manifest['inputs']:
        if item['group']=='additional' and not args.include_additional:continue
        dest=out/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            data=dest.read_bytes()
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Existing input differs; preserve and investigate: '+str(dest))
            done.append({'path':item['path'],'status':'existing_hash_verified'});continue
        request=urllib.request.Request(item['url'],headers={'User-Agent':'E3-frozen-scientific-replay/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response:data=response.read(item['bytes']+1)
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Downloaded immutable input differs: '+item['path'])
        with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as f:f.write(data);temporary=Path(f.name)
        os.replace(temporary,dest);dest.chmod(0o644);done.append({'path':item['path'],'status':'downloaded_and_hash_verified'})
    print(json.dumps({'status':'passed','inputs':done,'model_fits':0}))

if __name__=='__main__':main()
