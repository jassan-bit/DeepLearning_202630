"""Run a new container book/table case. CI is always labelled nonhuman.

Never invokes the historical C driver, training worker, or an ONNX runtime.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, importlib.util, json, math, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path
from prepare_external import SOURCE_COMMIT, METRICS, digest, inspect_kit, require
from reproduce_primary import FLOATS, FIELDS, regenerate, compare, check_published_table, read_table

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def validate_request(request,mode):
    require(request.get('source_commit')==SOURCE_COMMIT,'Request source edition differs')
    require(request.get('principal_table')=='V3-development-seven-recipes-section8.19','Scope must be explicit')
    require(request.get('fresh_training_requested') is False,'This entrypoint does not train')
    require(request.get('resources',{}).get('network_during_execution') is False,'Offline execution required')
    tol=request.get('tolerances');require(isinstance(tol,dict) and set(tol)==set(FLOATS),'Per-field tolerances absent')
    for values in tol.values():
        require(isinstance(values,dict) and set(values)=={'absolute','relative'},'Tolerance schema mismatch')
        require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values.values()),'Declare tolerances before execution; empty/null profiles do not authorize execution')
    if mode=='human-external':
        op=request.get('operator',{});require(request.get('mode')==mode,'Human request mode differs')
        require(all(isinstance(op.get(k),str) and op[k].strip() for k in ('name','contact_or_identifier','relationship_to_code')),'External operator must supply own identity and relationship')
        require(op.get('independent_of_code') is True,'Operator must explicitly attest external relationship')
        require(request.get('scope_accepted_by_operator') is True,'Operator must accept the exact principal-table scope')
        require(isinstance(request.get('scope_adequacy_explanation'),str) and request['scope_adequacy_explanation'].strip(),'Explain why the selected principal table meets the agreed scope')
        date=request.get('declared_before_execution_utc');require(isinstance(date,str),'Prior UTC declaration missing')
        d=datetime.datetime.fromisoformat(date.replace('Z','+00:00'));require(d.tzinfo is not None and d<=datetime.datetime.now(datetime.timezone.utc),'Prior declaration must be a past UTC timestamp')
    else:
        require(mode=='software-validation' and request.get('mode')==mode and request.get('human_execution_claimed') is False,'Automation must explicitly state human=false')
    return tol

def command(argv,log,timeout):
    with log.open('w') as f:
        completed=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=dict(os.environ,PYTHONHASHSEED='0',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1'),check=False)
    require(completed.returncode==0,'Command failed; evidence retained: '+str(argv)+' (see '+str(log)+')')

def run(case,mode):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic();case=Path(case).resolve();kit=Path(__file__).resolve().parent
    inspect_kit(kit);out=case/'output';source=case/'source'
    require(out.is_dir() and not list(out.iterdir()),'Output must be empty; use a new case for each execution')
    preparation=json.loads((case/'PREPARATION.json').read_text());inventory=json.loads((case/'SOURCE_INVENTORY.json').read_text())
    require(preparation['source_commit']==SOURCE_COMMIT and inventory['source_commit']==SOURCE_COMMIT,'Prepared source edition differs')
    require(digest(case/'SOURCE_INVENTORY.json')==preparation['source_inventory_sha256'],'Prepared inventory changed')
    profile_path=kit/'SOFTWARE_PROFILE.json' if mode=='software-validation' else case/'HUMAN_REQUEST.json'
    raw=profile_path.read_bytes();request=json.loads(raw);tolerances=validate_request(request,mode)
    write(out/'REQUEST_BEFORE_EXECUTION.json',request)
    write(out/'EXECUTION_STARTED.json',{'started_utc':started,'mode':mode,'profile_sha256':hashlib.sha256(raw).hexdigest(),'human_execution_claimed':False,'source_commit':SOURCE_COMMIT})
    try:
        for row in inventory['files']:
            p=source/row['path'];require(p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Prepared source changed before execution: '+row['path'])
        require((source/'requirements-book.lock.txt').read_bytes()==(kit/'requirements-book.lock.txt').read_bytes(),'Book dependency lock differs')
        wheel_report=Path('/opt/e3/pip-install-report.json')
        require(wheel_report.is_file(),'Declared Docker installation report missing; run this procedure in its container')
        shutil.copyfile(wheel_report,out/'pip-install-report.json')
        config=(source/'book_e3/_config.yml').read_text();require(re.search(r'execute_notebooks:\s*[\"\x27]?off[\"\x27]?',config),'Notebook execution must be explicitly off')
        require(not (source/'book_e3/_build').exists(),'A built book is already present')
        predictions=source/'experiments/e3_resolution_20261120/reproduction_C_preparation/references/assessment_predictions.csv'
        arithmetic=regenerate(predictions,out/'primary_V3_regenerated.csv')
        comparison=compare(kit/'references/primary_V3.csv',out/'primary_V3_regenerated.csv',tolerances);write(out/'primary_comparison.json',comparison)
        require(comparison['status']=='pass','Principal comparison failed; no tolerance adjustment is performed')
        published=check_published_table(source/'book_e3/sections/08_verificacion_estadistica.md',out/'primary_V3_regenerated.csv');write(out/'published_table_consistency.json',published)
        # Integrate the regenerated point-derived columns into the existing book
        # download. The archived CRPS_U column is preserved and labelled source,
        # never treated as recalculated from point predictions.
        path=source/METRICS
        with path.open(newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;original=list(reader)
        reconstructed=read_table(out/'primary_V3_regenerated.csv');require(len(original)==7,'Book download row count differs')
        preserved={r['name']:r.get('source_mean_CRPS_U_not_recomputed') for r in original}
        for row in original:
            require(row['name'] in reconstructed,'Book download recipe differs')
            row.update(reconstructed[row['name']])
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(original)
        require({r['name']:r.get('source_mean_CRPS_U_not_recomputed') for r in original}==preserved,'Stored source CRPS changed')
        write(out/'primary_regeneration.json',dict(arithmetic,published_table=published,book_download=METRICS,source_CRPS_U_preserved_not_recomputed=True))
        budget=request['resources'];limit=budget.get('timeout_seconds',1800);require(type(limit) is int and 60<=limit<=3600,'Timeout must be in60..3600seconds')
        command([sys.executable,'-m','pip','check'],out/'pip_check.log',120)
        command(['jupyter-book','build',str(source/'book_e3'),'--all','--warningiserror','--keep-going','--path-output',str(out/'book')],out/'book_build.log',limit)
        candidates=[out/'book/_build/html',out/'book/html'];html=next((p for p in candidates if (p/'intro.html').is_file()),None);require(html is not None,'Built HTML not found at declared output')
        spec=importlib.util.spec_from_file_location('pinned_link_checker',kit/'check_book_links.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        links=checker.check_html(html);write(out/'local_link_verification.json',dict(status='pass',**links))
        downloaded=html/'_static/e3_evidence_close_20261120/pilot/generated/metrics_recomputed.csv'
        require(downloaded.is_file() and downloaded.read_bytes()==path.read_bytes(),'Built book does not serve regenerated principal table')
        changes=[]
        for row in inventory['files']:
            p=source/row['path'];h=digest(p)
            if h!=row['sha256']:changes.append({'path':row['path'],'original_sha256':row['sha256'],'actual_sha256':h})
        require(all(r['path']==METRICS for r in changes),'Unexpected modification of tracked source in isolated case')
        installed=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json']))
        receipt={'status':'book_and_primary_table_software_passed','mode':mode,'source_commit':SOURCE_COMMIT,'started_utc':started,'elapsed_seconds':time.monotonic()-clock,
            'primary_table':'V3-development-seven-recipes-section8.19','scope':'Entire pinned E3 book build; regenerated V3 principal table only. No new fits or forward passes. Other precomputed tables are rendered, not claimed numerically regenerated.',
            'request_sha256_before_execution':hashlib.sha256(raw).hexdigest(),'profile_changed_during_execution':profile_path.read_bytes()!=raw,
            'comparison':comparison,'links':links,'book_html_relative':str(html.relative_to(case)),
            'regenerated_download_sha256':digest(downloaded),'isolated_source_changes':changes,
            'python':sys.version,'platform':platform.platform(),'installed_packages':installed,'pip_install_report_sha256':digest(out/'pip-install-report.json'),'new_model_fits':0,'new_model_predictions':0,
            'human_execution_claimed':False,'external_operator_request_present':mode=='human-external','human_attestation_still_required':True,'R2_or_R3_accredited_by_automation':False,
            'optional_C_executed':False,'historical_A_B_receipts_reused_not_rerun':True}
        require(not receipt['profile_changed_during_execution'],'Profile changed during execution')
        write(out/'EXECUTION_RECEIPT.json',receipt);return receipt
    except BaseException as e:
        write(out/'EXECUTION_FAILED.json',{'status':'failed_evidence_retained','exception_type':type(e).__name__,'message':str(e),'source_commit':SOURCE_COMMIT,'mode':mode,'human_execution_claimed':False,'new_model_fits':0,'new_model_predictions':0});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=Path('/case'));p.add_argument('--mode',choices=['software-validation','human-external'],required=True);a=p.parse_args()
    try:print(json.dumps(run(a.case,a.mode),sort_keys=True))
    except (ValueError,OSError,subprocess.SubprocessError) as e:print('Case blocked/failed: '+str(e),file=sys.stderr);raise SystemExit(2)
