"""Bounded independent HTTPS readback. No deploy, models, secrets or RNG."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SOURCE='551f6156c7b4d9a556bb37796176b2a49c354cb9'
PAGES='c0f49e5d3c680dd0e60c8a9393581a9edb7becb0'
SITE='https://jassan-bit.github.io/DeepLearning_202630/'
PATHS={'intro.html','entregable2/intro.html','entregable3/intro.html','entregables.html'}
LIMIT=10*1024*1024
HEADERS=['Date','Age','Cache-Control','ETag','Last-Modified','Content-Type',
         'Content-Encoding','Via','X-Cache','X-Served-By','X-GitHub-Request-Id','Location']


def sha(data):return hashlib.sha256(data).hexdigest()


def require(ok,text):
    if not ok:raise ValueError(text)


def validate_probe_url(url):
    parsed=urlsplit(url);site=urlsplit(SITE)
    allowed={site.path+p for p in PATHS|{'books_manifest.json'}}
    require((parsed.scheme,parsed.netloc)==(site.scheme,site.netloc)
            and parsed.path in allowed and not parsed.fragment,
            'URL is outside the pinned HTTPS publication')
    return parsed


def require_same_target(original,final):
    initial=validate_probe_url(original);target=validate_probe_url(final)
    require(initial.path==target.path,'response redirected to a different publication path')


class SamePublicationRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        require_same_target(req.full_url,newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def urlopen(request,timeout):
    return build_opener(SamePublicationRedirect()).open(request,timeout=timeout)


def validate_expected(expected):
    require(expected.get('source_commit')==SOURCE and expected.get('gh_pages_commit')==PAGES
            and expected.get('site_url')==SITE,'wrong expected source/site')
    m=expected['manifest'];require(m.get('source_commit')==SOURCE,'old expected manifest')
    require(len(m.get('books',[]))==3 and {b['entrypoint'] for b in m['books']}==PATHS-{'entregables.html'},
            'expected book routes differ')
    require(set(expected['page_sha256'])==PATHS,'four expected entrypoints required')
    for b in m['books']:require(expected['page_sha256'][b['entrypoint']]==b['intro_sha256'],'intro hash differs')


def fetch_once(url,manifest=False):
    """Worker process; outer subprocess timeout also bounds DNS/read trickles."""
    start=time.monotonic();record={'url':url}
    try:
        validate_probe_url(url)
        q=Request(url,headers={'Cache-Control':'no-cache, no-store, max-age=0','Pragma':'no-cache',
            'Accept-Encoding':'identity','User-Agent':'E3-independent-read-only-publication-probe'})
        with urlopen(q,timeout=8) as response:
            record.update(http_status=response.status,final_url=response.geturl(),
                          headers={k:response.headers[k] for k in HEADERS if k in response.headers})
            require_same_target(url,record['final_url'])
            wire=response.read(LIMIT+1)
        require(len(wire)<=LIMIT,'response exceeds limit');body=wire
        encoding=record['headers'].get('Content-Encoding','identity').lower()
        if encoding=='gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(wire)) as g:body=g.read(LIMIT+1)
        elif encoding not in {'','identity'}:raise ValueError('unsupported content encoding '+encoding)
        require(len(body)<=LIMIT,'decoded response exceeds limit')
        record.update(wire_bytes=len(wire),wire_sha256=sha(wire),body_bytes=len(body),body_sha256=sha(body))
        if manifest:record['manifest_object']=json.loads(body)
    except HTTPError as error:
        record.update(http_status=error.code,error_type='HTTPError',error=str(error),
                      headers={k:error.headers[k] for k in HEADERS if k in error.headers})
    except Exception as error:record.update(error_type=type(error).__name__,error=str(error))
    record['elapsed_seconds']=time.monotonic()-start
    return record


def bounded_request(url,manifest,timeout):
    command=[sys.executable,str(Path(__file__).resolve()),'--fetch-once',url]
    if manifest:command.append('--manifest-response')
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        if r.returncode:return {'url':url,'error_type':'worker_error','error':r.stderr[-500:]}
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {'url':url,'error_type':'hard_request_deadline','error':'request worker killed after deadline',
                'hard_deadline_seconds':timeout}
    except Exception as error:return {'url':url,'error_type':type(error).__name__,'error':str(error)}


def evaluate(expected,path,response):
    if response.get('http_status')!=200 or response.get('error_type'):
        return False,response.get('error','HTTP200 not received')
    if path=='books_manifest.json':
        ok=response.get('manifest_object')==expected['manifest']
        return ok,'manifest_exact' if ok else 'manifest differs from pinned artifact'
    ok=response.get('body_sha256')==expected['page_sha256'][path]
    return ok,'page_hash_exact' if ok else 'page SHA256 differs from pinned artifact'


def probe(expected,report,attempts=3,timeout=10,interval=5,request=bounded_request,sleep=time.sleep):
    validate_expected(expected)
    require(1<=attempts<=3 and 0<timeout<=10 and 0<=interval<=5,'probe exceeds bounded budget')
    report=Path(report);start=time.monotonic()
    result={'kind':'independent_HTTPS_readback_not_deployment_or_scientific_validation',
            'expected_source_commit':SOURCE,'expected_gh_pages_commit':PAGES,'site_url':SITE,
            'attempt_limit':attempts,'per_request_hard_deadline_seconds':timeout,
            'interval_seconds':interval,'attempts':[],'web_deployment_verified':False,
            'GitHub_or_deployment_writes':False,'new_forecasting_model_fits':0,
            'new_market_predictions':0,'new_statistical_resamples':0,'R3_accredited':False}
    for number in range(1,attempts+1):
        # UTC wall-clock nonce only: no random/statistical sampling.
        nonce=f'{time.time_ns()}-{number}'
        observations=[];success=True
        for path in ['books_manifest.json','intro.html','entregable2/intro.html','entregable3/intro.html','entregables.html']:
            url=urljoin(SITE,path)+'?source_check='+SOURCE+'&independent_probe='+nonce
            response=request(url,path=='books_manifest.json',timeout)
            ok,reason=evaluate(expected,path,response)
            if path=='books_manifest.json' and isinstance(response.get('manifest_object'),dict):
                response['served_source_commit']=response['manifest_object'].get('source_commit')
            response.pop('manifest_object',None)
            item={'path':path,'match':ok,'comparison':reason,**response};observations.append(item)
            print(json.dumps({'attempt':number,**item},ensure_ascii=False),flush=True);success=success and ok
        result['attempts'].append({'number':number,'fresh_nonce':nonce,'all_five_responses_match':success,
                                   'observations':observations})
        result['elapsed_seconds']=time.monotonic()-start
        if success:
            result.update(web_deployment_verified=True,verified_source_commit=SOURCE,
                          verified_entrypoints=['intro.html','entregable2/intro.html','entregable3/intro.html','entregables.html'],
                          verified_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        if success:return result
        if number<attempts:sleep(interval)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--attempts',type=int,default=3);p.add_argument('--request-timeout',type=float,default=10)
    p.add_argument('--interval',type=float,default=5);p.add_argument('--fetch-once');p.add_argument('--manifest-response',action='store_true')
    a=p.parse_args()
    if a.fetch_once:print(json.dumps(fetch_once(a.fetch_once,a.manifest_response)),flush=True)
    else:
        require(a.expected is not None and a.report is not None,'expected artifact and report required')
        out=probe(json.loads(a.expected.read_text()),a.report,a.attempts,a.request_timeout,a.interval)
        print(json.dumps({'publication_readback_receipt':out},ensure_ascii=False),flush=True)
        sys.exit(0 if out['web_deployment_verified'] else 1)
