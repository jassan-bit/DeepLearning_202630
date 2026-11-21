"""Publish the verified scoped review capsule to the owner's GitHub repository.

This helper does not write Zenodo, change model weights, or grant licenses.
Only exact locally verified files are uploaded, without clobbering assets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError,URLError
from urllib.parse import quote,urlsplit
from urllib.request import HTTPRedirectHandler,Request,build_opener

REPO='jassan-bit/DeepLearning_202630'
TAG='e3-source-evidence-8d03a73-v0.3.1-review'
HOSTS={'github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'}

def require(ok,message):
    if not ok:raise ValueError(message)

def gh(args,check=True):
    r=subprocess.run(['gh',*args],capture_output=True,text=True)
    if check and r.returncode:raise RuntimeError('GitHub CLI operation failed: '+args[0])
    return r

def allowed(url):
    p=urlsplit(url)
    require(p.scheme=='https' and p.hostname in HOSTS and not p.username and not p.password
            and p.port in (None,443),'Unexpected release download origin')
    return url

class Redirects(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,message,headers,url):
        allowed(url)
        return super().redirect_request(request,fp,code,message,headers,url)

def _public_digest_once(url,wanted):
    request=Request(allowed(url),headers={'User-Agent':'E3-scoped-archive-public-download-check/0.3.1'})
    h=hashlib.sha256();n=0
    with build_opener(Redirects()).open(request,timeout=60) as stream:
        require(stream.status==200,'Public release download not HTTP200')
        final_url=stream.url
        while True:
            chunk=stream.read(1024*1024)
            if not chunk:break
            n+=len(chunk);require(n<=wanted['bytes'],'Public asset exceeds declared bound');h.update(chunk)
    require(n==wanted['bytes'] and h.hexdigest()==wanted['sha256'],'Public release asset differs')
    return {'http':200,'bytes':n,'sha256':h.hexdigest(),'public_URL':url,'download_host':urlsplit(final_url).hostname}

def public_digest(url,wanted):
    for attempt in range(4):
        try:return _public_digest_once(url,wanted)
        except HTTPError as error:
            retry_after=error.headers.get('Retry-After') if error.headers else None
            if (error.code not in (404,429,500,502,503,504) and not (error.code==403 and retry_after)) or attempt==3:
                raise
            if retry_after:
                try:delay=float(retry_after)
                except ValueError:delay=3*(2**attempt)
                if delay>60:raise
            else:delay=3*(2**attempt)
        except (URLError,TimeoutError):
            if attempt==3:raise
            delay=3*(2**attempt)
        time.sleep(max(0,delay))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();root=args.directory.resolve()
    launcher=os.environ.get('GITHUB_SHA','');require(bool(re.fullmatch('[0-9a-f]{40}',launcher)),'Pinned launcher commit absent')
    report={'status':'not_published','repository':REPO,'tag':TAG,'launcher_commit':launcher,
            'source_repository_cut':'8d03a73d529eb3f9b2e89992b4488cfeb88a99e6','prerelease_REVIEW':True,
            'Zenodo_writes':0,'new_model_fits':0,'R3_accredited':False,'global_license_granted':False}
    try:
        receipt=json.loads((root/'consolidation_receipt.json').read_text())
        require(receipt.get('status') == 'scoped_archive_consolidated_with15_exact_approved_ONNX',
                'Executed consolidation receipt not passed')
        require(receipt.get('source_commit')=='8d03a73d529eb3f9b2e89992b4488cfeb88a99e6'
                and receipt.get('version')=='0.3.1-REVIEW' and receipt.get('R3_accredited') is False
                and receipt.get('full_historical_master_deposited') is False,
                'Unexpected consolidation scope')
        require(receipt.get('verification',{}).get('unique_ONNX_graphs_verified')==15
                and receipt.get('verification',{}).get('all_member_hashes_verified') is True
                and receipt.get('C_addon',{}).get('included') is True,
                'Final model identity, extraction or C preparation check absent')
        capsule=receipt.get('capsule',receipt.get('archive',{}));name=capsule.get('file') or capsule.get('name')
        require(isinstance(name,str) and Path(name).name==name and name.endswith('.tar.gz'),'Unsafe capsule name')
        files=[root/name,root/'capsule_manifest.json',root/'consolidation_receipt.json',root/'RELEASE_NOTES.md']
        assets={}
        for f in files:
            require(f.is_file() and not f.is_symlink(),'Release input absent/symlink')
            raw=f.read_bytes();require(len(raw)<=200_000_000,'Release file exceeds cap')
            assets[f.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        require(assets[name]['bytes']==capsule['bytes'] and assets[name]['sha256']==capsule['sha256'],'Archive receipt differs')
        require(sum(v['bytes'] for v in assets.values())<=220_000_000,'Release total exceeds bound')
        view=gh(['api','repos/'+REPO+'/releases/tags/'+TAG],check=False)
        if view.returncode:
            gh(['release','create',TAG,'--repo',REPO,'--target',launcher,'--prerelease',
                '--title','E3 source and evidence capsule 0.3.1 — REVIEW','--notes-file',str(root/'RELEASE_NOTES.md')])
            existing=json.loads(gh(['api','repos/'+REPO+'/releases/tags/'+TAG]).stdout)
        else:existing=json.loads(view.stdout)
        require(existing.get('prerelease') is True and existing.get('draft') is False,'Unexpected release state')
        tag=json.loads(gh(['api','repos/'+REPO+'/git/ref/tags/'+TAG]).stdout)
        require(tag['object']['type']=='commit' and tag['object']['sha']==launcher,'Existing tag points to different source')
        by_name={a['name']:a for a in existing.get('assets',[])}
        require(set(by_name)<=set(assets),'Unexpected existing release asset; preserve and review')
        for filename,wanted in assets.items():
            if filename in by_name:
                require(by_name[filename]['size']==wanted['bytes'],'Existing asset size differs; no overwrite')
                public_digest('https://github.com/'+REPO+'/releases/download/'+TAG+'/'+quote(filename),wanted)
            else:gh(['release','upload',TAG,str(root/filename),'--repo',REPO])
        checks={}
        for filename,wanted in assets.items():
            checks[filename]=public_digest('https://github.com/'+REPO+'/releases/download/'+TAG+'/'+quote(filename),wanted)
        report.update(status='review_release_published_and_public_asset_hashes_verified',
                      release_url='https://github.com/'+REPO+'/releases/tag/'+TAG,assets=checks,
                      publication_scope='permitted source/evidence capsule with explicit exclusions; not full master or new Zenodo deposit')
    except Exception as error:report['error']={'type':type(error).__name__,'message':str(error)[:700]}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('E3_REVIEW_RELEASE_RECEIPT_BEGIN');print(json.dumps(report,ensure_ascii=False,indent=2));print('E3_REVIEW_RELEASE_RECEIPT_END')
    return 0 if report['status']=='review_release_published_and_public_asset_hashes_verified' else 2

if __name__=='__main__':raise SystemExit(main())
