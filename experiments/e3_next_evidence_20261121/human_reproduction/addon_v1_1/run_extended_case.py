"""Run comprehensive19x4+V3 saved-result reconstruction and clean E3 book. CI is nonhuman.

Never invokes the historical C driver, training worker, or an ONNX runtime.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, importlib.util, json, math, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path
from html.parser import HTMLParser
HERE=Path(__file__).resolve().parent
BASE_KIT=HERE.parent if (HERE.parent/'prepare_external.py').is_file() else HERE.parent/'kit'
BASE_MANIFEST_SHA256='a35a5f56d6288ad5640f347b59415f00968277dceb96096fda0dc83db75b0206'
sys.path.insert(0,str(BASE_KIT))
from audit_primary import regenerate as regenerate_audit, validate_tolerances as validate_audit_tolerances, PREFIX as AUDIT_PREFIX
from prepare_external import SOURCE_COMMIT, METRICS, digest, inspect_kit, require
from reproduce_primary import FLOATS, FIELDS, regenerate, compare, check_published_table, read_table

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def validate_request(request,mode):
    require(request.get('source_commit')==SOURCE_COMMIT,'Request source edition differs')
    require(request.get('principal_table')=='additive19x4-clean-plus-development-V3','Scope must be explicit')
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
    validate_audit_tolerances(request.get('audit_tolerances'))
    return tol

def command(argv,log,timeout):
    with log.open('w') as f:
        completed=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=dict(os.environ,PYTHONHASHSEED='0',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1'),check=False)
    require(completed.returncode==0,'Command failed; evidence retained: '+str(argv)+' (see '+str(log)+')')

def verify_generated_html(path):
    class Tables(HTMLParser):
        def __init__(self):super().__init__();self.depth=0;self.active=False;self.start_depth=None;self.rows=[];self.table_depth=0
        def handle_starttag(self,tag,attrs):
            if tag=='section':self.depth+=1
            if dict(attrs).get('id')=='e3-human-generated-principal':self.active=True;self.start_depth=self.depth
            if self.active and tag=='table':self.rows.append(0);self.table_depth+=1
            if self.active and tag=='tr' and self.table_depth:self.rows[-1]+=1
        def handle_endtag(self,tag):
            if tag=='table' and self.active:self.table_depth-=1
            if tag=='section':
                if self.active and self.depth==self.start_depth:self.active=False
                self.depth-=1
    parser=Tables();parser.feed(Path(path).read_text());require(parser.rows==[77,8],'Built generated appendix must render76 additive and7 V3 rows plus headers')
    return {'status':'pass','tables':2,'rendered_additive_rows':76,'rendered_V3_rows':7,'headers_excluded':True}

def append_generated_tables(chapter,out,source):
    """Render genuinely regenerated rows while retaining every historical blob."""
    old=chapter.read_text();require('(e3-human-generated-principal)=' not in old,'Generated appendix already exists')
    lines=['','', '(e3-human-generated-principal)=','## Reproducción local de las tablas principales', '',
        'Suplemento generado durante esta ejecución del contenedor a partir de resultados costosos versionados. Se declara asistencia de IA en el generador; esta sección no acredita una persona externa ni confirmación.', '',
        'La tabla aditiva se reagrega desde4060 métricas por semilla:76filas de prueba limpia desde580registros, más532agregados completos y456rangos. No se recalculan las métricas desde1874560predicciones ni se repiten ajustes, inferencias o remuestreos. El piloto usa935predicciones existentes. Los CSV y manifiestos históricos originales conservan todos sus bytes y hashes; estos derivados tienen rutas y recibos separados.', '',
        '{download}`76 filas de la maestra19×4 <../_static/human_reproduction_19x4/principal_additive_19x4.csv>` · {download}`532 agregados <../_static/human_reproduction_19x4/model_metrics_regenerated.csv>` · {download}`456 rangos <../_static/human_reproduction_19x4/cross_dataset_ranks_regenerated.csv>` · {download}`resumen4conjuntos <../_static/human_reproduction_19x4/clean_test_summary_regenerated.csv>` · {download}`pilotoV3 <../_static/human_reproduction_19x4/metrics_V3_regenerated.csv>`', '',
        '### Maestra aditiva completa de prueba limpia', '',
        '| Conjunto | Sistema | Semillas/trayectorias | Fechas | RMSE media individual | DE RMSE | MAE media | R² medio | Puesto descriptivo |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    with (out/'audit/principal_additive_19x4.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==76,'Generated appendix must retain all76main rows')
    for r in rows:
        sd='No aplica' if r['RMSE_seed_sample_SD']=='' else f"{float(r['RMSE_seed_sample_SD']):.9f}"
        lines.append(f"| {r['dataset']} | {r['model']} | {r['seed_count']} | {r['dates']} | {float(r['RMSE_mean_individual']):.9f} | {sd} | {float(r['MAE_mean_individual']):.9f} | {float(r['R2_mean_individual']):.9f} | {float(r['rank_RMSE']):g}/19 |")
    lines.extend(['','### Piloto V3 de desarrollo','',
        'No constituye confirmación. atom_crps conserva el resultado adverso y NO-GO; no se promociona automáticamente la ablación ganadora. CRPS_U sigue etiquetado como fuente no recalculada.', '',
        '| Receta | Trayectorias | Fechas | RMSE media individual | DE RMSE | MAE media |', '|---|---:|---:|---:|---:|---:|'])
    for name,r in read_table(out/'primary_V3_regenerated.csv').items():
        sd='No aplica' if r['SD_individual_RMSE']=='' else f"{float(r['SD_individual_RMSE']):.9f}"
        lines.append(f"| {name} | {r['trajectories']} | {r['dates_per_trajectory']} | {float(r['mean_individual_RMSE']):.9f} | {sd} | {float(r['mean_individual_MAE']):.9f} |")
    chapter.write_text(old+'\n'.join(lines)+'\n')

def run(case,mode):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic();case=Path(case).resolve();kit=BASE_KIT
    require(digest(kit/'MANIFEST.json')==BASE_MANIFEST_SHA256,'Frozen base-v1 manifest differs')
    inspect_kit(kit);inspect_kit(HERE);out=case/'output';source=case/'source'
    require(out.is_dir() and not list(out.iterdir()),'Output must be empty; use a new case for each execution')
    preparation=json.loads((case/'PREPARATION.json').read_text());inventory=json.loads((case/'SOURCE_INVENTORY.json').read_text())
    require(preparation['source_commit']==SOURCE_COMMIT and inventory['source_commit']==SOURCE_COMMIT,'Prepared source edition differs')
    require(digest(case/'SOURCE_INVENTORY.json')==preparation['source_inventory_sha256'],'Prepared inventory changed')
    profile_path=HERE/'SOFTWARE_PROFILE.json' if mode=='software-validation' else case/'HUMAN_REQUEST.json'
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
        original_path=source/METRICS
        path=source/'book_e3/_static/human_reproduction_19x4/metrics_V3_regenerated.csv'
        path.parent.mkdir(parents=True,exist_ok=True)
        with original_path.open(newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;original=list(reader)
        reconstructed=read_table(out/'primary_V3_regenerated.csv');require(len(original)==7,'Book download row count differs')
        preserved={r['name']:r.get('source_mean_CRPS_U_not_recomputed') for r in original}
        for row in original:
            require(row['name'] in reconstructed,'Book download recipe differs')
            row.update(reconstructed[row['name']])
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(original)
        require({r['name']:r.get('source_mean_CRPS_U_not_recomputed') for r in original}==preserved,'Stored source CRPS changed')
        write(out/'primary_regeneration.json',dict(arithmetic,published_table=published,book_download=path.relative_to(source).as_posix(),source_CRPS_U_preserved_not_recomputed=True))
        audit=regenerate_audit(source,out/'audit',request['audit_tolerances'],install_in_book=True)
        chapter=source/'book_e3/sections/08_verificacion_estadistica.md'
        append_generated_tables(chapter,out,source)
        budget=request['resources'];limit=budget.get('timeout_seconds',1800);require(type(limit) is int and 60<=limit<=3600,'Timeout must be in60..3600seconds')
        command([sys.executable,'-m','pip','check'],out/'pip_check.log',120)
        command(['jupyter-book','build',str(source/'book_e3'),'--all','--warningiserror','--keep-going','--path-output',str(out/'book')],out/'book_build.log',limit)
        candidates=[out/'book/_build/html',out/'book/html'];html=next((p for p in candidates if (p/'intro.html').is_file()),None);require(html is not None,'Built HTML not found at declared output')
        spec=importlib.util.spec_from_file_location('pinned_link_checker',kit/'check_book_links.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        links=checker.check_html(html);write(out/'local_link_verification.json',dict(status='pass',**links))
        downloaded=html/'_static/human_reproduction_19x4/metrics_V3_regenerated.csv'
        require(downloaded.is_file() and downloaded.read_bytes()==path.read_bytes(),'Built book does not serve regenerated principal table')
        served_audit=[]
        for relative in audit['installed_generated_paths']:
            built=html/relative.removeprefix('book_e3/');require(built.is_file() and built.read_bytes()==(source/relative).read_bytes(),'Built book does not serve regenerated additive audit table: '+relative)
            served_audit.append({'path':relative,'sha256':digest(built)})
        main76=html/'_static/human_reproduction_19x4/principal_additive_19x4.csv'
        require(main76.is_file() and digest(main76)==audit['main76_csv_sha256'],'Built book lacks reconstructed main19x4 table')
        generated_html=verify_generated_html(html/'sections/08_verificacion_estadistica.html');write(out/'generated_appendix_HTML_verification.json',generated_html)
        changes=[]
        for row in inventory['files']:
            p=source/row['path'];h=digest(p)
            if h!=row['sha256']:changes.append({'path':row['path'],'original_sha256':row['sha256'],'actual_sha256':h})
        require(all(r['path']=='book_e3/sections/08_verificacion_estadistica.md' for r in changes),'Unexpected modification of tracked source in isolated case')
        installed=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json']))
        receipt={'status':'book_additive19x4_and_V3_software_passed','mode':mode,'source_commit':SOURCE_COMMIT,'started_utc':started,'elapsed_seconds':time.monotonic()-clock,
            'primary_table':'additive19x4-clean-plus-development-V3','scope':'Entire pinned E3 book build; additive532aggregate rows including76clean-main rows,456ranks and4dataset summary reaggregated from4060fixed cached per-seed metrics and2320interval summaries; V3 seven-recipe table recomputed from935saved points. No per-seed recomputation from1874560audit points, resampling, fits, forward pass, confirmatory result or human claim.',
            'request_sha256_before_execution':hashlib.sha256(raw).hexdigest(),'profile_changed_during_execution':profile_path.read_bytes()!=raw,
            'comparison':comparison,'audit_regeneration':audit,'served_regenerated_audit_tables':served_audit,'base_v1_manifest_sha256':BASE_MANIFEST_SHA256,'addon_manifest_sha256':digest(HERE/'MANIFEST.json'),'links':links,'book_html_relative':str(html.relative_to(case)),
            'regenerated_download_sha256':digest(downloaded),'isolated_source_changes':changes,'original_CSVs_and_manifests_byte_unchanged':True,'generated_appendix_HTML_verification':generated_html,
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
