"""Acceptance client for the opt-in historical14 supplement, no model imports.

Requires an exact service-reported deployment SHA before any prediction request.
This automated engineering replay never constitutes independent confirmation,
training reproduction or cryptographic attestation of remote source code.
"""
from __future__ import annotations
import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

HERE=Path(__file__).resolve().parent
GRAPH_IDS=('eda-series','eda-hist','eda-feature','eda-partitions-figure','eda-acf','eda-pacf')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None


def utcnow():return datetime.now(timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def origin(url, allow_local):
    p=urlsplit(url)
    require(not p.username and not p.password and not p.query and not p.fragment and p.path in ('','/'), 'Supply an origin URL only')
    local=p.hostname in ('localhost','127.0.0.1','::1')
    require((p.scheme=='https' and not local) or (allow_local and local and p.scheme=='http'),'Use HTTPS or explicit --allow-local')
    return url.rstrip('/'),not local


def request(base,path,body=None):
    raw=None if body is None else json.dumps(body,allow_nan=False).encode()
    req=Request(base+path,data=raw,headers={'User-Agent':'E3-profile14-acceptance/1',**({'Content-Type':'application/json'} if raw is not None else {})})
    started=time.perf_counter()
    try:
        with build_opener(NoRedirect()).open(req,timeout=30) as response:
            status=response.status;payload=response.read(1024*1024+1)
    except HTTPError as error:
        status=error.code;payload=error.read(1024*1024+1)
    require(len(payload)<=1024*1024,'Response exceeds bounded acceptance client limit')
    require(not 300<=status<400,'Unexpected redirect rejected')
    try:document=json.loads(payload)
    except (UnicodeError,json.JSONDecodeError):document=None
    return status,document,(time.perf_counter()-started)*1000


def shifted_csv(text,target):
    rows=list(csv.DictReader(io.StringIO(text)));out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=['date','open','high','low','close','volume'],lineterminator='\n');writer.writeheader()
    for i,row in enumerate(rows):row['date']=(date.fromisoformat(target)-timedelta(days=7-i)).isoformat();writer.writerow(row)
    return out.getvalue()


def ood_csv(text):
    rows=list(csv.DictReader(io.StringIO(text)));out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=['date','open','high','low','close','volume'],lineterminator='\n');writer.writeheader()
    for row in rows:row['volume']='1e30';writer.writerow(row)
    return out.getvalue()


def check_prediction(d,m,reference):
    require(isinstance(d,dict),'Prediction body missing')
    require(d['model_id']==m['id'] and d['seed']==m['seed'],'Wrong representative')
    require(d['model_sha256']==m['model_sha256'] and d['metadata_sha256']==m['metadata_sha256'],'Wrong artifact fingerprints')
    require(d['profile_sha256']==reference['profile_sha256'],'Wrong profile fingerprint')
    require(math.isfinite(d['prediction']) and abs(d['prediction']-m['prediction'])<=reference['absolute_tolerance_sigma7'],'Frozen point parity failed')
    require(d['calibration_q']==m['q_sigma7'] and d['nominal_coverage']==.9,'Frozen quantile or nominal coverage changed')
    expected_kind='posterior_E3_frozen_profile' if m['id']=='CNN-LSTM-atencion-aditiva E3' else 'E2_frozen_original'
    require(d['calibration_kind']==expected_kind,'Calibration provenance kind changed')
    if expected_kind=='posterior_E3_frozen_profile':require(d['calibration_profile']=='posterior_absolute_residual_90_v2','Wrong E3 posterior calibration profile')
    expected=[max(0,d['prediction']-m['q_sigma7']),d['prediction']+m['q_sigma7']]
    require(d['interval']==expected,'Interval formula changed')
    e=d['explanation'];require(e['method']=='occlusion_raw_window' and d['explanation_status']=='sensitivity_not_causal','Explanation scope changed')
    require([x['lag'] for x in e['attributions']]==list(range(-7,0)),'Missing lag explanation')
    require(all(math.isfinite(x['effect']) and math.isfinite(x['prediction_with_reference_bar']) for x in e['attributions']),'Nonfinite explanation')
    require(math.isfinite(d['latency_ms']) and d['latency_ms']>=0 and math.isfinite(e['latency_ms']) and e['latency_ms']>=0,'Invalid timing')
    require(type(d['ood']) is bool and 0<=d['ood_fraction']<=1,'Invalid OOD status')
    require(d['new_training_runs']==d['new_calibration_fits']==0,'Unexpected fitting claim')


def browser_checks(base,args,reference,report):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        options={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
        if args.chromium_path:options['executable_path']=args.chromium_path
        browser=pw.chromium.launch(**options)
        context=browser.new_context(viewport={'width':1440,'height':1000},locale='es-CO',timezone_id='America/Bogota',service_workers='block')
        def route_guard(route):
            p=urlsplit(route.request.url)
            if p.scheme in ('http','https') and (p.scheme,p.netloc)!=(urlsplit(base).scheme,urlsplit(base).netloc):
                report['blocked_cross_origin_requests'].append(p.hostname);route.abort('blockedbyclient')
            else:route.continue_()
        context.route('**/*',route_guard)
        page=context.new_page();page.set_default_timeout(30000)
        page.on('pageerror',lambda error:report['browser_errors'].append(str(error)))
        page.on('console',lambda message:report['console_errors'].append(message.text) if message.type=='error' else None)
        report['chromium_version']=browser.version
        try:
            for path in ('/','/eda','/modelos'):
                started=time.perf_counter();response=page.goto(base+path,wait_until='networkidle');require(response and response.status==200,'Legacy navigation failed')
                page.locator('main h1').wait_for()
                callback_checked=False
                if path=='/eda':
                    page.wait_for_function("ids=>ids.every(id=>{const g=document.querySelector('#'+id+' .js-plotly-plot');return g&&g.layout&&g.layout.meta&&g.layout.meta.filtered_n===1316;})",arg=list(GRAPH_IDS))
                    page.locator('#eda-partitions .Select-clear-zone').click()
                    page.wait_for_function("ids=>ids.every(id=>{const g=document.querySelector('#'+id+' .js-plotly-plot');return g&&g.layout&&g.layout.meta&&g.layout.meta.filtered_n===0&&g.data.length===0;})",arg=list(GRAPH_IDS));callback_checked=True
                if path=='/modelos':page.wait_for_function('document.querySelectorAll(\'#leaders-table td[data-dash-column="model"]\').length===17')
                report['navigation'].append({'path':path,'HTTP':response.status,'elapsed_ms_including_callbacks':(time.perf_counter()-started)*1000,'EDA_clear_callback_checked':callback_checked})
                if path=='/modelos':screenshot(page,args.screenshots_dir,'original_five_ui.jpg',report)
            response=page.goto(base+'/perfil-14',wait_until='networkidle');require(response and response.status==200,'Supplement page failed')
            page.wait_for_function('document.querySelectorAll("#model option").length===14')
            require(page.locator('#model option').all_text_contents()==[m['id'] for m in reference['models']],'Browser selector differs from frozen reference')
            page.locator('#csv').fill(reference['case']['csv_text']);page.locator('#target').fill(reference['case']['target_date'])
            for m in reference['models']:
                page.locator('#model').select_option(m['id']);started=time.perf_counter()
                with page.expect_response(lambda r:r.url==base+'/api/profile14/predict' and r.request.method=='POST') as pending:
                    page.locator('#submit').click()
                require(pending.value.status==200,'Browser prediction callback failed');d=pending.value.json();check_prediction(d,m,reference)
                page.locator('#result h2').wait_for();page.wait_for_function('document.querySelector("#status").textContent==="Pronóstico calculado con artefactos verificados."')
                require(m['id'] in page.locator('#result h2').inner_text(),'Wrong rendered model')
                require(page.locator('#result table tr').count()==8,'Missing seven rendered explanation rows')
                require(d['model_sha256'] in page.locator('#fingerprints').inner_text(),'Missing rendered artifact fingerprint')
                report['browser_predictions'].append({'model':m['id'],'HTTP':200,'elapsed_ms_including_HTTP_render_and_explanation':(time.perf_counter()-started)*1000,'rendered_explanation_rows':7})
            screenshot(page,args.screenshots_dir,'profile14_desktop.jpg',report)
            page.set_viewport_size({'width':390,'height':844});screenshot(page,args.screenshots_dir,'profile14_mobile.jpg',report)
            require(not report['browser_errors'] and not report['console_errors'] and not report['blocked_cross_origin_requests'],'Browser errors or cross-origin requests occurred')
        finally:context.close();browser.close()


def screenshot(page,directory,name,report):
    destination=Path(directory)/name;destination.parent.mkdir(parents=True,exist_ok=True);page.screenshot(path=str(destination),full_page=not name.startswith('original_five'),animations='disabled',type='jpeg',quality=55);raw=destination.read_bytes();report['screenshots'].append({'name':name,'sha256':sha(raw),'bytes':len(raw)})


def run(args):
    base,public=origin(args.url,args.allow_local)
    require(re.fullmatch('[0-9a-f]{40}',args.expected_sha or ''),'--expected-sha requires an exact 40-digit lowercase commit')
    reference_raw=Path(args.reference).read_bytes();reference=json.loads(reference_raw)
    report={'schema_version':1,'started_at_utc':utcnow(),'status':'running','base_url':base,'scope':'public_acceptance' if public else 'local_container_acceptance_not_Render','declared_expected_commit':args.expected_sha,'identity_basis':'Service-reported environment SHA matched to operator-declared source; not cryptographic attestation.','client_python':platform.python_version(),'client_sha256':sha(Path(__file__).read_bytes()),'reference_sha256':sha(reference_raw),'API_predictions':[],'interval_date_guards':[],'OOD_cases':[],'invalid_requests':[],'navigation':[],'browser_predictions':[],'browser_errors':[],'console_errors':[],'blocked_cross_origin_requests':[],'screenshots':[],'new_fits':0,'new_calibrations':0,'public_Render_acceptance_passed':False,'limits':['Engineering replays on a fixed historical window and explicitly shifted/adversarial input fixtures, not scientific new dates.','A reported deployment SHA is not proof of every remote byte.','Local execution cannot establish public Render acceptance, remote RAM or uptime.','Automated acceptance is not human independent training reproduction, R1/R2/R3 or a prospective statistical campaign.'],'errors':[]}
    try:
        status,health,elapsed=request(base,'/healthz');report['health']={'HTTP':status,'body':health,'elapsed_ms':elapsed}
        require(status==200 and isinstance(health,dict) and health.get('status')=='ok','Service not ready')
        require(health.get('source_commit')==args.expected_sha and health.get('source_commit_source') in ('APP_SOURCE_COMMIT','RENDER_GIT_COMMIT'),'Identity unavailable or SHA mismatch: no prediction requests issued')
        require(health.get('profile_predictors')==14 and health.get('original_UI_predictors')==5 and health.get('V3_enabled') is False,'Wrong product profile')
        require(health.get('profile_sha256')==reference['profile_sha256'],'Profile fingerprint mismatch')
        status,cat,_=request(base,'/api/profile14/catalogue');require(status==200 and cat['V3_enabled'] is False,'Catalogue unavailable')
        require([x['id'] for x in cat['models']]==[x['id'] for x in reference['models']],'Catalogue model mismatch')
        report['deployment_identity_matched']=True
        for m in reference['models']:
            body={'model_id':m['id'],'csv':reference['case']['csv_text'],'target_date':reference['case']['target_date']}
            status,d,elapsed=request(base,'/api/profile14/predict',body);require(status==200,'Prediction failed');check_prediction(d,m,reference)
            report['API_predictions'].append({'model':m['id'],'HTTP':status,'prediction':d['prediction'],'frozen_prediction':m['prediction'],'absolute_error':abs(d['prediction']-m['prediction']),'q_sigma7':d['calibration_q'],'latency_ms_excludes_explanation':d['latency_ms'],'explanation_ms':d['explanation']['latency_ms'],'HTTP_elapsed_ms':elapsed,'model_sha256':d['model_sha256'],'metadata_sha256':d['metadata_sha256']})
            shifted={**body,'csv':shifted_csv(body['csv'],'2024-12-30'),'target_date':'2024-12-30'}
            status,d,elapsed=request(base,'/api/profile14/predict',shifted);require(status==200 and d['interval'] is None and d['calibration_q'] is None and d['nominal_coverage'] is None,'Posterior interval applied at calibration boundary')
            report['interval_date_guards'].append({'model':m['id'],'HTTP':status,'target_date':'2024-12-30','fixture_dates_shifted':True,'interval_disabled':True})
            status,d,elapsed=request(base,'/api/profile14/predict',{**body,'csv':ood_csv(body['csv'])});require(status==200 and d['ood'] is True and d['ood_fraction']>0,'Missing OOD warning')
            report['OOD_cases'].append({'model':m['id'],'HTTP':status,'volume_1e30_fixture':True,'ood':d['ood'],'ood_fraction':d['ood_fraction']})
        valid={'model_id':'MLP16','csv':reference['case']['csv_text'],'target_date':reference['case']['target_date']}
        fixtures=[('V3_blocked',{**valid,'model_id':'V3 atom_crps experimental'}),('classic_blocked',{**valid,'model_id':'Persistence'}),('model_type',{**valid,'model_id':[]}),('date_type',{**valid,'target_date':{}}),('csv_type',{**valid,'csv':[]}),('bad_csv',{**valid,'csv':'bad'}),('oversized_csv',{**valid,'csv':'x'*102401}),('missing_field',{'model_id':'MLP16'}),('extra_field',{**valid,'unexpected':1}),('not_object',[])]
        for name,body in fixtures:
            status,d,_=request(base,'/api/profile14/predict',body);require(status==400 and isinstance(d,dict) and isinstance(d.get('error'),str),'Invalid input not safely rejected')
            require('Traceback' not in d['error'] and '/workspace' not in d['error'],'Internal detail leaked')
            report['invalid_requests'].append({'fixture':name,'HTTP':status,'safe_error':d['error']})
        if not args.no_browser:browser_checks(base,args,reference,report)
        report['status']='passed';report['public_Render_acceptance_passed']=public and urlsplit(base).hostname=='deeplearning-e3-dashboard.onrender.com' and not args.no_browser
        report['public_deployment_acceptance_passed']=public and not args.no_browser
    except Exception as error:
        report['status']='failed';report['errors'].append({'type':type(error).__name__,'message':str(error)[:1500]})
    report['finished_at_utc']=utcnow();Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--allow-local',action='store_true');p.add_argument('--reference',default=str(HERE/'acceptance_reference.json'));p.add_argument('--output',required=True);p.add_argument('--screenshots-dir',required=True);p.add_argument('--chromium-path');p.add_argument('--no-browser',action='store_true');args=p.parse_args();r=run(args);print(json.dumps({'status':r['status'],'scope':r['scope'],'API_predictions':len(r['API_predictions']),'browser_predictions':len(r['browser_predictions']),'output':args.output}));raise SystemExit(0 if r['status']=='passed' else 1)
