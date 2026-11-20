"""Download only the pilot weights already public at commit8eaa581; no inference.

AI assistance declared. This script does not export, train, calibrate, upload,
modify a Git reference, or download any newly derived ONNX artifact.
"""
from pathlib import Path
import argparse,hashlib,json,os,tempfile,urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet-root',required=True,type=Path)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('public_weight_locators.json'))
    args=p.parse_args();root=args.packet_root.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());verified=[]
    if manifest['unique_checkpoints']!=15 or len(manifest['files'])!=16:raise ValueError('Expected15 frozen checkpoints and one representative duplicate')
    for item in manifest['files']:
        if item['commit']!='8eaa581bc34b615a8507c9e384a3e4be29084720':raise ValueError('Unexpected source commit')
        expected_url='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+item['commit']+'/'+item['source_path']
        if item['url']!=expected_url or not item['source_path'].startswith('runs/e3_v3_development_pilot/20261120T031424176339Z/payload/jobs/final/') or not item['source_path'].endswith('/supervised/checkpoint.pt'):raise ValueError('Unexpected source path')
        dest=(root/item['destination']).resolve()
        if not dest.is_relative_to(root) or dest.is_symlink():raise ValueError('Unsafe output path')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():data=dest.read_bytes();status='existing_verified'
        elif item.get('duplicate_of'):
            source=(root/item['duplicate_of']).resolve()
            if not source.is_relative_to(root):raise ValueError('Unsafe duplicate source')
            data=source.read_bytes();status='copied_verified_original_public_checkpoint'
        else:
            request=urllib.request.Request(item['url'],headers={'User-Agent':'E3-public-frozen-weight-hydration/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response:data=response.read(item['bytes']+1)
            status='downloaded_original_public_checkpoint'
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Checkpoint hash/length differs; preserve existing destination: '+item['destination'])
        if not dest.exists():
            with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as f:f.write(data);temporary=Path(f.name)
            os.replace(temporary,dest);dest.chmod(0o644)
        verified.append({'destination':item['destination'],'status':status,'sha256':item['sha256'],'bytes':len(data)})
    print(json.dumps({'status':'passed','unique_original_public_checkpoints':15,'verified_destinations':len(verified),'files':verified,'new_model_fits':0,'new_model_inferences':0,'ONNX_exports':0,'uploads':0}))


if __name__=='__main__':main()
