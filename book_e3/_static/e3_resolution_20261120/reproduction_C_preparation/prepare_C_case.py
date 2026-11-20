"""Copy pinned sources/references read-only; optionally acquire the pinned input."""
import argparse
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler,Request,build_opener
from historical_C_driver import HOME,MANIFEST_SHA,atomic_json,require,safe,sha,no_symlinks

class Redirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,url):
        p=urlsplit(url);require(p.scheme=='https' and p.hostname=='raw.githubusercontent.com','Unexpected input redirect')
        return super().redirect_request(req,fp,code,msg,headers,url)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case-root',type=Path,required=True)
    parser.add_argument('--input-file',type=Path);parser.add_argument('--fetch-input',action='store_true');parser.add_argument('--max-download-bytes',type=int,default=500000)
    args=parser.parse_args();case=args.case_root.absolute();no_symlinks(case);require(not case.exists(),'New case directory required')
    require(not (args.input_file and args.fetch_input),'Choose local verified input or explicit download')
    m=HOME/'SOURCE_MANIFEST.json';require(sha(m)==MANIFEST_SHA,'Prepared manifest changed');manifest=json.loads(m.read_bytes())
    input_root=case/'input';input_root.mkdir(parents=True)
    for row in manifest['source_files']:
        source=HOME/safe(row['path']);require(sha(source)==row['sha256'] and source.stat().st_size==row['bytes'],'Prepared source/reference changed')
        target=input_root/safe(row['path']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copyfile(m,input_root/'SOURCE_MANIFEST.json')
    wanted=manifest['input'];target=input_root/'data/cohort.npz'
    if args.input_file:
        require(args.input_file.stat().st_size==wanted['bytes'] and sha(args.input_file)==wanted['sha256'],'Local input SHA differs')
        target.parent.mkdir(exist_ok=True);shutil.copyfile(args.input_file,target)
    elif args.fetch_input:
        require(args.max_download_bytes>=wanted['bytes'],'Explicit input download ceiling too small')
        u=urlsplit(wanted['url']);require(u.scheme=='https' and u.hostname=='raw.githubusercontent.com' and not u.username and not u.password,'Unexpected input origin')
        with build_opener(Redirects()).open(Request(wanted['url'],headers={'User-Agent':'E3-historical-C-case-preparer/1'}),timeout=60) as response:
            require(response.status==200,'Input HTTP failed');raw=response.read(wanted['bytes']+1)
        require(len(raw)==wanted['bytes'],'Downloaded input length differs');target.parent.mkdir(exist_ok=True);target.write_bytes(raw)
        require(sha(target)==wanted['sha256'],'Downloaded input SHA differs')
    for p in sorted(input_root.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    input_root.chmod(0o555);shutil.copyfile(HOME/'C_REQUEST.template.json',case/'C_REQUEST.template.json')
    atomic_json(case/'preparation_receipt.json',{'status':'C_case_prepared_without_runtime_import_or_training',
       'source_files_verified':len(manifest['source_files']),'input_acquired':target.exists(),
       'source_manifest_sha256':MANIFEST_SHA,'C_training_executed':False,'GPU_used':False,
       'human_execution_observed':False,'R3_accredited':False})
    print(str(case/'preparation_receipt.json'));return 0

if __name__=='__main__':raise SystemExit(main())
